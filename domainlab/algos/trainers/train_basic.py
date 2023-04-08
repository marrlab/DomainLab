"""
basic trainer
"""
import math
from torch import optim

from domainlab.algos.trainers.a_trainer import AbstractTrainer
from domainlab.algos.trainers.a_trainer import mk_opt


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
        self.epo_loss_tr = 0
        for ind_batch, (tensor_x, vec_y, vec_d, *others) in enumerate(self.loader_tr):
            tensor_x, vec_y, vec_d = \
                tensor_x.to(self.device), vec_y.to(self.device), vec_d.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model.cal_loss(tensor_x, vec_y, vec_d, others)
            loss = loss.sum()
            loss.backward()
            self.optimizer.step()
            self.epo_loss_tr += loss.detach().item()
            self.after_batch(epoch, ind_batch)
        assert self.epo_loss_tr is not None
        assert not math.isnan(self.epo_loss_tr)
        flag_stop = self.observer.update(epoch)  # notify observer
        assert flag_stop is not None
        return flag_stop

    def train_batch(self, tensor_x, vec_y, vec_d, others):
        """
        use a temporary optimizer to update only the model upon a batch of data
        """
        # temparary optimizer
        optimizer = mk_opt(self.model, self.aconf)
        tensor_x, vec_y, vec_d = \
            tensor_x.to(self.device), vec_y.to(self.device), vec_d.to(self.device)
        optimizer.zero_grad()
        loss = self.model.cal_loss(tensor_x, vec_y, vec_d, others)
        loss = loss.sum()
        loss.backward()
        optimizer.step()
