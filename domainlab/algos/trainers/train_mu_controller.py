"""
update hyper-parameters during training
"""
from operator import add
import torch
from domainlab.algos.trainers.train_basic import TrainerBasic
from domainlab.algos.trainers.fbopt_alternate import HyperSchedulerFeedbackAlternave
from domainlab.utils.logger import Logger


def list_divide(list_val, scalar):
    return [ele/scalar for ele in list_val]

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
        counter = 0.0
        with torch.no_grad():
            for _, (tensor_x, vec_y, vec_d, *_) in enumerate(self.loader_tr_no_drop):
                tensor_x, vec_y, vec_d = \
                    tensor_x.to(self.device), vec_y.to(self.device), vec_d.to(self.device)
                tuple_reg_loss = self.model.cal_reg_loss(tensor_x, vec_y, vec_d)
                # NOTE: first [0] extract the loss, second [0] get the list
                list_b_reg_loss = tuple_reg_loss[0]
                list_b_reg_loss_sumed = [ele.sum().item() for ele in list_b_reg_loss]
                if len(epo_reg_loss) == 0:
                    epo_reg_loss = list_b_reg_loss_sumed
                else:
                    epo_reg_loss = list(map(add, epo_reg_loss, list_b_reg_loss_sumed))
                b_task_loss = self.model.cal_task_loss(tensor_x, vec_y).sum()
                # sum will kill the dimension of the mini batch
                epo_task_loss += b_task_loss
                counter += 1.0
        return list_divide(epo_reg_loss, counter), epo_task_loss/counter

    def before_batch(self, epoch, ind_batch):
        """
        if hyper-parameters should be updated per batch, then step
        should be set to epoch*self.num_batches + ind_batch
        """
        if self.flag_update_hyper_per_batch:
            # NOTE: if not update per_batch, then not updated
            self.model.hyper_update(epoch*self.num_batches + ind_batch, self.hyper_scheduler)
        return super().after_batch(epoch, ind_batch)

    def before_tr(self):
        self.set_scheduler(scheduler=HyperSchedulerFeedbackAlternave)
        self.model.hyper_update(epoch=None, fun_scheduler=HyperSetter(self.hyper_scheduler.mmu))
        self.epo_reg_loss_tr, self.epo_task_loss_tr = self.eval_r_loss()
        self.hyper_scheduler.set_setpoint(
            [ele * self.aconf.ini_setpoint_ratio for ele in self.epo_reg_loss_tr],
            self.epo_task_loss_tr)

    def tr_epoch(self, epoch):
        """
        update hyper-parameters only per epoch
        """
        flag = super().tr_epoch(epoch)
        self.hyper_scheduler.search_mu(
            self.epo_reg_loss_tr,
            self.epo_task_loss_tr,
            dict(self.model.named_parameters()),
            miter=epoch)
        self.hyper_scheduler.update_setpoint(self.epo_reg_loss_tr, self.epo_task_loss_tr)
        return flag