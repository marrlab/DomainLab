"""
basic trainer
"""
import math
from operator import add

import torch

from domainlab import g_tensor_batch_agg
from domainlab.algos.trainers.a_trainer import AbstractTrainer, mk_opt


def list_divide(list_val, scalar):
    return [ele / scalar for ele in list_val]


class TrainerBasic(AbstractTrainer):
    """
    basic trainer
    """

    def before_tr(self):
        """
        check the performance of randomly initialized weight
        """
        self.model.evaluate(self.loader_te, self.device)

    def before_epoch(self):
        """
        set model to train mode
        initialize some member variables
        """
        self.model.train()
        self.counter_batch = 0.0
        self.epo_loss_tr = 0
        self.epo_reg_loss_tr = [0.0 for _ in range(10)]
        self.epo_task_loss_tr = 0

    def tr_epoch(self, epoch):
        self.before_epoch()
        for ind_batch, (tensor_x, tensor_y, tensor_d, *others) in enumerate(
            self.loader_tr
        ):
            self.tr_batch(tensor_x, tensor_y, tensor_d, others, ind_batch, epoch)
        return self.after_epoch(epoch)

    def after_epoch(self, epoch):
        """
        observer collect information
        """
        self.epo_loss_tr /= self.counter_batch
        self.epo_task_loss_tr /= self.counter_batch
        self.epo_reg_loss_tr = list_divide(self.epo_reg_loss_tr, self.counter_batch)
        assert self.epo_loss_tr is not None
        assert not math.isnan(self.epo_loss_tr)
        flag_stop = self.observer.update(epoch)  # notify observer
        assert flag_stop is not None
        return flag_stop

    def log_loss(self, list_b_reg_loss, loss_task, loss):
        """
        just for logging the self.epo_reg_loss_tr
        """
        self.epo_task_loss_tr += loss_task.sum().detach().item()
        #
        list_b_reg_loss_sumed = [ele.sum().detach().item() for ele in list_b_reg_loss]
        self.epo_reg_loss_tr = list(
            map(add, self.epo_reg_loss_tr, list_b_reg_loss_sumed)
        )
        self.epo_loss_tr += loss.detach().item()

    def tr_batch(self, tensor_x, tensor_y, tensor_d, others, ind_batch, epoch):
        """
        optimize neural network one step upon a mini-batch of data
        """
        self.before_batch(epoch, ind_batch)
        tensor_x, tensor_y, tensor_d = (
            tensor_x.to(self.device),
            tensor_y.to(self.device),
            tensor_d.to(self.device),
        )
        self.optimizer.zero_grad()
        loss = self.cal_loss(tensor_x, tensor_y, tensor_d, others)
        loss.backward()
        self.optimizer.step()
        self.after_batch(epoch, ind_batch)
        self.counter_batch += 1

    def cal_loss(self, tensor_x, tensor_y, tensor_d, others):
        """
        so that user api can use trainer.cal_loss to train
        """
        loss_task = self.model.cal_task_loss(tensor_x, tensor_y)

        list_reg_tr_batch, list_mu_tr = self.cal_reg_loss(
            tensor_x, tensor_y, tensor_d, others
        )
        tensor_batch_reg_loss_penalized = self.model.list_inner_product(
            list_reg_tr_batch, list_mu_tr
        )
        assert len(tensor_batch_reg_loss_penalized.shape) == 1
        loss_erm_agg = g_tensor_batch_agg(loss_task)
        loss_reg_penalized_agg = g_tensor_batch_agg(tensor_batch_reg_loss_penalized)
        loss_penalized = (
            self.model.multiplier4task_loss * loss_erm_agg + loss_reg_penalized_agg
        )
        self.log_loss(list_reg_tr_batch, loss_task, loss_penalized)
        return loss_penalized
