"""
basic trainer
"""
import math
from operator import add

from domainlab.algos.trainers.a_trainer import AbstractTrainer
from domainlab.algos.trainers.a_trainer import mk_opt


def list_divide(list_val, scalar):
    return [ele/scalar for ele in list_val]


class TrainerBasic(AbstractTrainer):
    """
    basic trainer
    """
    def before_tr(self):
        """
        check the performance of randomly initialized weight
        """
        self.model.evaluate(self.loader_te, self.device)

    def tr_epoch(self, epoch):
        self.model.train()
        self.counter_batch = 0.0
        self.epo_loss_tr = 0
        self.epo_reg_loss_tr = [0.0 for _ in range(10)]
        self.epo_task_loss_tr = 0
        for ind_batch, (tensor_x, tensor_y, tensor_d, *others) in enumerate(self.loader_tr):
            self.tr_batch(tensor_x, tensor_y, tensor_d, others, ind_batch, epoch)
        self.epo_loss_tr /= self.counter_batch
        self.epo_task_loss_tr /= self.counter_batch
        self.epo_reg_loss_tr = list_divide(self.epo_reg_loss_tr, self.counter_batch)
        assert self.epo_loss_tr is not None
        assert not math.isnan(self.epo_loss_tr)
        flag_stop = self.observer.update(epoch)  # notify observer
        assert flag_stop is not None
        return flag_stop

    def handle_r_loss(self, list_b_reg_loss):
        """
        just for logging
        """
        list_b_reg_loss_sumed = [ele.sum().detach().item() for ele in list_b_reg_loss]
        self.epo_reg_loss_tr = list(map(add, self.epo_reg_loss_tr, list_b_reg_loss_sumed))
        return list_b_reg_loss_sumed

    def cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        """
        decorate trainer regularization loss
        """
        return self._cal_reg_loss(tensor_x, tensor_y, tensor_d, others)

    def _cal_reg_loss(tensor_x, tensor_y, tensor_d, others):
        """
        trainer specific regularization loss, by default 0
        """
        device = tensor_x.device
        bsize = tensor_x.shape[0]
        return [torch.zeros(bsize, 1).to(device)], [0.0]

    def tr_batch(self, tensor_x, tensor_y, tensor_d, others, ind_batch, epoch):
        """
        different from self.train_batch(...), which is used for mldg, the current function
        is used inside tr_epoch
        """
        self.before_batch(epoch, ind_batch)
        tensor_x, tensor_y, tensor_d = \
            tensor_x.to(self.device), tensor_y.to(self.device), tensor_d.to(self.device)
        self.optimizer.zero_grad()
        loss, list_loss_reg, loss_task = self.model.cal_loss(tensor_x, tensor_y, tensor_d, others)
        #
        list_reg_tr, list_mu_tr = self.cal_reg_loss(tensor_x, tensor_y, tensor_d, others)
        #
        self.handle_r_loss(list_loss_reg)
        reg_tr = self.model.inner_product(list_reg_tr, list_mu_tr)
        loss = loss.sum() +  reg_tr.sum()
        loss.backward()
        self.optimizer.step()
        self.epo_loss_tr += loss.detach().item()
        self.epo_task_loss_tr += loss_task.sum().detach().item()
        self.after_batch(epoch, ind_batch)
        self.counter_batch += 1

    def train_batch(self, tensor_x, tensor_y, tensor_d, others):
        """
        use a temporary optimizer to update only the model upon a batch of data
        """
        # temparary optimizer
        optimizer = mk_opt(self.model, self.aconf)
        tensor_x, tensor_y, tensor_d = \
            tensor_x.to(self.device), tensor_y.to(self.device), tensor_d.to(self.device)
        optimizer.zero_grad()
        loss, *_ = self.model.cal_loss(tensor_x, tensor_y, tensor_d, others)
        loss = loss.sum()
        loss.backward()
        optimizer.step()
