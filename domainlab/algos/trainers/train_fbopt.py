
"""
feedback optimization
"""
import copy
from operator import add

import torch

from domainlab.algos.trainers.a_trainer import AbstractTrainer
from domainlab.algos.trainers.train_basic import TrainerBasic
from domainlab.algos.trainers.fbopt import HyperSchedulerFeedback
from domainlab.algos.trainers.fbopt_alternate import HyperSchedulerFeedbackAlternave
from domainlab.algos.msels.c_msel_bang import MSelBang
from domainlab.utils.logger import Logger


class HyperSetter():
    """
    mock object to force hyper-parameter in the model
    """
    def __init__(self, dict_hyper):
        self.dict_hyper = dict_hyper

    def __call__(self, epoch=None):
        return self.dict_hyper


class TrainerFbOpt(AbstractTrainer):
    """
    feedback optimization
    """
    def set_scheduler(self, scheduler=HyperSchedulerFeedback):
        """
        Args:
            scheduler: The class name of the scheduler, the object corresponding to
            this class name will be created inside model
        """
        # model.hyper_init will register the hyper-parameters of the model to scheduler
        self.hyper_scheduler = self.model.hyper_init(scheduler, trainer=self)

    def before_tr(self):
        """
        before training begins, construct helper objects
        """
        if self.aconf.msel == "last":
            self.observer.model_sel = MSelBang(max_es=None)
            # although validation distribution will be very difference from test distribution, it is still a better
            # idea to not to use the last iteration of the model
        # self.set_scheduler(scheduler=HyperSchedulerFeedback)
        self.set_scheduler(scheduler=HyperSchedulerFeedbackAlternave)
        self.model.evaluate(self.loader_te, self.device)
        self.inner_trainer = TrainerBasic()  # look ahead
        # here we need a mechanism to generate deep copy of the model
        self.inner_trainer.init_business(
            copy.deepcopy(self.model), self.task, self.observer, self.device, self.aconf,
            flag_accept=False)

        epo_reg_loss, epo_task_loss = self.eval_r_loss()
        self.hyper_scheduler.set_setpoint(
            [ele * self.aconf.ini_setpoint_ratio for ele in epo_reg_loss], epo_task_loss)

    def opt_theta(self, dict4mu, dict_theta0):
        """
        operator for theta, move gradient for one epoch, then check if criteria is met
        this method will be invoked by the hyper-parameter scheduling object
        """
        self.inner_trainer.reset()
        self.inner_trainer.model.set_params(dict_theta0)
        # mock the model hyper-parameter to be from dict4mu
        self.inner_trainer.model.hyper_update(epoch=None, fun_scheduler=HyperSetter(dict4mu))
        # hide implementation details of inner_trainer
        self.inner_trainer.model.train()
        for _, (tensor_x, vec_y, vec_d, *others) in enumerate(self.inner_trainer.loader_tr):
            self.inner_trainer.train_batch(tensor_x, vec_y, vec_d, others)  # update inner_net
        dict_par = dict(self.inner_trainer.model.named_parameters())
        return dict_par

    def eval_p_loss(self, dict4mu, dict_theta):
        """
        evaluate the penalty function value on all available training data
        # TODO: normalize loss via batchsize
        """
        temp_model = copy.deepcopy(self.model)
        # mock the model hyper-parameter to be from dict4mu
        temp_model.hyper_update(epoch=None, fun_scheduler=HyperSetter(dict4mu))
        temp_model.set_params(dict_theta)
        epo_p_loss = 0  # penalized loss
        with torch.no_grad():
            for _, (tensor_x, vec_y, vec_d, *_) in enumerate(self.loader_tr_no_drop):
                tensor_x, vec_y, vec_d = \
                    tensor_x.to(self.device), vec_y.to(self.device), vec_d.to(self.device)
                # sum will kill the dimension of the mini batch
                b_p_loss = temp_model.cal_loss(tensor_x, vec_y, vec_d).sum()
                epo_p_loss += b_p_loss
        return epo_p_loss

    def eval_r_loss(self):
        """
        evaluate the regularization loss and ERM loss with respect ot parameter dict_theta
        ERM loss on all available training data
        # TODO: normalize loss via batchsize
        """
        temp_model = copy.deepcopy(self.model)
        temp_model.eval()
        # mock the model hyper-parameter to be from dict4mu
        epo_reg_loss = []
        epo_task_loss = 0
        with torch.no_grad():
            for _, (tensor_x, vec_y, vec_d, *_) in enumerate(self.loader_tr_no_drop):
                tensor_x, vec_y, vec_d = \
                    tensor_x.to(self.device), vec_y.to(self.device), vec_d.to(self.device)
                tuple_reg_loss = temp_model.cal_reg_loss(tensor_x, vec_y, vec_d)
                # NOTE: first [0] extract the loss, second [0] get the list
                list_b_reg_loss = tuple_reg_loss[0]   # FIXME: this only works when scalar multiplier
                list_b_reg_loss_sumed = [ele.sum().item() for ele in list_b_reg_loss]
                if len(epo_reg_loss) == 0:
                    epo_reg_loss = list_b_reg_loss_sumed
                else:
                    epo_reg_loss = list(map(add, epo_reg_loss, list_b_reg_loss_sumed))
                # FIXME: change this to vector
                # each component of vector is a mini batch loss
                b_task_loss = temp_model.cal_task_loss(tensor_x, vec_y).sum()
                # sum will kill the dimension of the mini batch
                epo_task_loss += b_task_loss
        return epo_reg_loss, epo_task_loss

    def tr_epoch(self, epoch):
        """
        the algorithm will try to search for the reg-descent operator, only when found,
        the model will tunnel/jump/shoot into the found pivot parameter $\\theta^{(k+1)}$,
        otherwise,
        """
        epo_reg_loss, epo_task_loss = self.eval_r_loss()
        if self.aconf.msel == "loss_tr":
            if self.aconf.msel_tr_loss =="reg":
                self.epo_loss_tr = epo_reg_loss
            elif self.aconf.msel_tr_loss =="task":
                self.epo_loss_tr = epo_task_loss
            else:
                raise RuntimeError("msel_tr_loss set to be the wrong value")
        elif self.aconf.msel == "last" or self.aconf.msel == "val":
            self.epo_loss_tr = 1.0 # FIXME: check if this is not used at all
        else:
            raise RuntimeError("msel type not supported")

        logger = Logger.get_logger(logger_name='main_out_logger', loglevel="INFO")
        logger.info(f"at epoch {epoch}, epo_reg_loss={epo_reg_loss}, epo_task_loss={epo_task_loss}")

        # double check if loss evaluation is the same when executed two times
        epo_reg_loss, epo_task_loss = self.eval_r_loss()

        logger = Logger.get_logger(logger_name='main_out_logger', loglevel="INFO")
        logger.info(f"at epoch {epoch}, epo_reg_loss={epo_reg_loss}, epo_task_loss={epo_task_loss}")

        # self.model.train()  # FIXME: i guess no need to put into train mode?

        flag_success = self.hyper_scheduler.search_mu(
            dict(self.model.named_parameters()),
            miter=epoch)  # FIXME: iter_start=0 or 1?

        if flag_success:
            # only in success case, mu will be updated
            logger.info("pivot parameter found, jumping/shooting there now!")
            epo_reg_loss, epo_task_loss = self.eval_r_loss()
            logger = Logger.get_logger(logger_name='main_out_logger', loglevel="INFO")
            logger.info(
                f"at epoch {epoch}, before shooting: epo_reg_loss={epo_reg_loss},  \
                epo_task_loss={epo_task_loss}")

            # shoot/tunnel to new found parameter configuration
            self.model.set_params(self.hyper_scheduler.dict_theta)

            epo_reg_loss, epo_task_loss = self.eval_r_loss()
            logger = Logger.get_logger(logger_name='main_out_logger', loglevel="INFO")
            logger.info(
                f"at epoch {epoch}, after shooting: epo_reg_loss={epo_reg_loss}, \
                epo_task_loss={epo_task_loss}")
            self.hyper_scheduler.update_setpoint(epo_reg_loss, epo_task_loss)
                #if self.aconf.myoptic_pareto:
                #    self.hyper_scheduler.update_anchor(dict_par)
        flag_early_stop_observer = self.observer.update(epoch)
        flag_msel_early_stop = self.observer.model_sel.if_stop()
        self.mu_iter_start = 1   # start from mu=0, due to arange(iter_start, budget)
        # FIXME: after a long seesion with Emilio, we could not resovle this, this is a hack at themoment
        return flag_msel_early_stop
