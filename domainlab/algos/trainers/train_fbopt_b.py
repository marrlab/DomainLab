"""
update hyper-parameters during training
"""
from operator import add
import torch
from domainlab.algos.trainers.train_basic import TrainerBasic
from domainlab.algos.trainers.fbopt_mu_controller import HyperSchedulerFeedback
from domainlab.algos.trainers.hyper_scheduler import HyperSchedulerWarmup
from domainlab.utils.logger import Logger


def list_divide(list_val, scalar):
    """
    divide a list by a scalar
    """
    return [ele / scalar for ele in list_val]


class HyperSetter():
    """
    mock object to force hyper-parameter in the model
    """
    def __init__(self, dict_hyper):
        self.dict_hyper = dict_hyper

    def __call__(self, epoch=None):
        return self.dict_hyper


class TrainerFbOpt(TrainerBasic):
    """
    TrainerHyperScheduler
    """
    def set_scheduler(self, scheduler):
        """
        Args:
            scheduler: The class name of the scheduler, the object corresponding to
            this class name will be created inside model
        """
        # model.hyper_init will register the hyper-parameters of the model to scheduler
        self.hyper_scheduler = self.model.hyper_init(scheduler, trainer=self)

    def eval_r_loss(self):
        """
        evaluate the regularization loss and ERM loss with respect ot parameter dict_theta
        ERM loss on all available training data
        # TODO: normalize loss via batchsize
        """
        self.model.eval()
        # mock the model hyper-parameter to be from dict4mu
        epo_reg_loss = []
        epo_task_loss = 0
        epo_p_loss = 0
        counter = 0.0
        with torch.no_grad():
            for _, (tensor_x, vec_y, vec_d, *others) in enumerate(self.loader_tr_no_drop):
                tensor_x, vec_y, vec_d = \
                    tensor_x.to(self.device), vec_y.to(self.device), vec_d.to(self.device)
                tuple_reg_loss = self.model.cal_reg_loss(tensor_x, vec_y, vec_d, others)
                p_loss, *_ = self.model.cal_loss(tensor_x, vec_y, vec_d, others)
                # NOTE: first [0] extract the loss, second [0] get the list
                list_b_reg_loss = tuple_reg_loss[0]
                list_b_reg_loss_sumed = [ele.sum().detach().item() for ele in list_b_reg_loss]
                if len(epo_reg_loss) == 0:
                    epo_reg_loss = list_b_reg_loss_sumed
                else:
                    epo_reg_loss = list(map(add, epo_reg_loss, list_b_reg_loss_sumed))
                b_task_loss = self.model.cal_task_loss(tensor_x, vec_y).sum().detach().item()
                # sum will kill the dimension of the mini batch
                epo_task_loss += b_task_loss
                epo_p_loss += p_loss.sum().detach().item()
                counter += 1.0
        return list_divide(epo_reg_loss, counter), epo_task_loss / counter, epo_p_loss / counter

    def before_batch(self, epoch, ind_batch):
        """
        if hyper-parameters should be updated per batch, then step
        should be set to epoch*self.num_batches + ind_batch
        """
        if self.flag_update_hyper_per_batch:
            # NOTE: if not update per_batch, then not updated
            self.model.hyper_update(epoch * self.num_batches + ind_batch, self.hyper_scheduler)
        return super().after_batch(epoch, ind_batch)

    def before_tr(self):
        self.flag_setpoint_updated = False
        if self.aconf.force_feedforward:
            self.set_scheduler(scheduler=HyperSchedulerWarmup)
        else:
            self.set_scheduler(scheduler=HyperSchedulerFeedback)

        self.set_model_with_mu()  # very small value
        if self.aconf.tr_with_init_mu:
            self.tr_with_init_mu()

        self.epo_reg_loss_tr, self.epo_task_loss_tr, self.epo_loss_tr = self.eval_r_loss()
        self.hyper_scheduler.set_setpoint(
            [ele * self.aconf.ini_setpoint_ratio if ele > 0 else
             ele / self.aconf.ini_setpoint_ratio for ele in self.epo_reg_loss_tr],
            self.epo_task_loss_tr)  # setpoing w.r.t. random initialization of neural network
        self.hyper_scheduler.set_k_i_gain(self.epo_reg_loss_tr)

    @property
    def list_str_multiplier_na(self):
        """
        return the name of multipliers
        """
        return self.model.list_str_multiplier_na

    def tr_with_init_mu(self):
        """
        erm step with very small mu
        """
        super().tr_epoch(-1)

    def set_model_with_mu(self):
        """
        set model multipliers
        """
        self._model.hyper_update(epoch=None, fun_scheduler=HyperSetter(self.hyper_scheduler.mmu))

    def tr_epoch(self, epoch, flag_info=False):
        """
        update multipliers only per epoch
        """
        self.hyper_scheduler.search_mu(
            self.epo_reg_loss_tr,
            self.epo_task_loss_tr,
            self.epo_loss_tr,
            self.list_str_multiplier_na,
            miter=epoch)
        self.set_model_with_mu()
        if hasattr(self.model, "dict_multiplier"):
            logger = Logger.get_logger()
            logger.info(f"current multiplier: {self.model.dict_multiplier}")

        flag = super().tr_epoch(epoch, self.flag_setpoint_updated)
        # is it good to update setpoint after we know the new value of each loss?
        self.flag_setpoint_updated = self.hyper_scheduler.update_setpoint(
            self.epo_reg_loss_tr, self.epo_task_loss_tr)
        return flag
