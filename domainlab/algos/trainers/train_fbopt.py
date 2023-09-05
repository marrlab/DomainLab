"""
feedback optimization
"""
import copy

from domainlab.algos.trainers.a_trainer import AbstractTrainer
from domainlab.algos.trainers.train_basic import TrainerBasic
from domainlab.algos.trainers.fbopt import HyperSchedulerFeedback


class HyperSetter():
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

    def before_tr(self):
        """
        check the performance of randomly initialized weight
        """
        self.set_scheduler()
        self.model.evaluate(self.loader_te, self.device)
        self.inner_trainer = TrainerBasic()
        self.inner_trainer.init_business(
            self.model, self.task, self.observer, self.device, self.aconf,
            flag_accept=False)

    def opt_theta(self, dict4mu, theta0):
        """
        operator for theta, move gradient for one epoch, then check if criteria is met
        """
        self.inner_trainer.model.set_params(theta0)   # FIXME: implement for each model
        self.inner_trainer.model.hyper_update(epoch=None, fun_scheduler=HyperSetter(dict4mu))
        for _, (tensor_x, vec_y, vec_d, *others) in enumerate(self.inner_trainer.loader_tr):
            self.inner_trainer.train_batch(tensor_x, vec_y, vec_d, others)  # update inner_net
            # the following is not needed anymore
            # loss_look_forward = inner_net.cal_task_loss(tensor_x, vec_y)
            # loss_source = self.model.cal_loss(tensor_x, vec_y, vec_d, others)
            # loss = loss_source.sum() + self.aconf.gamma_reg * loss_look_forward.sum()
            # loss.backward()
            # self.optimizer.step()
        dict_par = self.inner_trainer.model.name_parameters()
        return dict_par

    def eval_loss(self, dict4mu, theta):
        """
        evaluate the penalty function value
        """
        temp_model = copy.deepcopy(self.model)
        temp_model.hyper_update(epoch=None, fun_scheduler=HyperSetter(dict4mu))
        temp_model.set_params(theta) # FIXME: for each model, implement net1.load_state_dict(net2.state_dict())
        epo_reg_loss = 0
        epo_task_loss = 0
        epo_p_loss = 0  # penalized loss
        # FIXME: will loader be corupted? if called at different places?
        for ind_batch, (tensor_x, vec_y, vec_d, *_) in enumerate(self.loader_tr):
            b_reg_loss = temp_model.cal_reg_loss(tensor_x, vec_y).sum()
            b_task_loss = temp_model.cal_task_loss(tensor_x, vec_y).sum()  # sum will kill the dimension of the mini batch
            b_p_loss = temp_model.cal_p_loss(tensor_x, vec_y).sum()
            epo_reg_loss += b_reg_loss
            epo_task_loss += b_task_loss
            epo_p_loss += b_p_loss
        return epo_p_loss

    def tr_epoch(self, epoch):
        self.model.train()
        self.hyper_scheduler.search_mu()   # if mu not found, will raise error
        self.model.set_params(self.hyper_scheduler.theta)
        flag_stop = self.observer.update(epoch)  # notify observer
        return flag_stop
