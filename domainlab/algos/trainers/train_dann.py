import torch
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

from domainlab.algos.trainers.a_trainer import TrainerClassif


class TrainerDANN(TrainerClassif):
    def __init__(self, model, observer, device,
                 loader_tr=None, loader_te=None, lr=None, aconf=None, task=None):
        super().__init__(model, observer, device, loader_tr, loader_te, lr, aconf, task)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=5e-4, weight_decay=0)
        self.lr0 = 5e-4
        self.y_criterion = nn.NLLLoss()
        self.d_criterion = nn.NLLLoss()
        self._alpha = 0.1
        self._current_step = None
        self._n_steps = None
        self._annealing_func = None
        self._lr_scheduler = None
        self.set_alpha_scheduler(n_epochs * len(train_loader), annealing_func='exp')
        self.set_lr_scheduler(self.optimizer, n_epochs * len(train_loader), self.lr0)


    def set_alpha_scheduler(self, n_steps, annealing_func='exp'):
        self._current_step = 0
        self._n_steps = n_steps
        self._annealing_func = annealing_func

    def alpha_scheduler_step(self):
        self._current_step += 1

    def set_lr_scheduler(self, optimizer, n_steps, lr0, lamb=None):
        if lamb is None:
            lamb = lambda current_step: lr0 / ((1 + 10 * (current_step / n_steps)) ** 0.75)
        scheduler = LambdaLR(optimizer, lr_lambda=[lamb])
        self._lr_scheduler = scheduler

    def lr_scheduler_step(self):
        self._lr_scheduler.step()

    def scheduler_step(self):
        if self._annealing_func is not None:
            self.alpha_scheduler_step()
        if self._lr_scheduler is not None:
            self.lr_scheduler_step()

    @property
    def _current_alpha(self):
        if self._annealing_func is None:
            return self._alpha
        elif self._annealing_func == 'exp':
            p = float(self._current_step) / self._n_steps
            return float((2. / (1. + np.exp(-10 * p)) - 1) * self._alpha)
        else:
            raise Exception()

    def tr_epoch(self, epoch):
        itr_per_epoch = len(train_loader)
        n_epochs = 10000 // itr_per_epoch
        self.set_alpha_scheduler(n_epochs * len(train_loader), annealing_func='exp')
        self.set_lr_scheduler(self.optimizer, n_epochs * len(train_loader), self.lr0)

        for _, (tensor_x, vec_y, vec_d) in enumerate(self.loader_tr):
            tensor_x, vec_y, vec_d = \
                tensor_x.to(self.device), vec_y.to(self.device), vec_d.to(self.device)
            self.optimizer.zero_grad()
            self.scheduler_step()  # learning rate and alpha together
            y_pred, d_pred = self.model(tensor_x, self._current_alpha)
            y_loss = self.y_criterion(y_pred, torch.max(y, 1)[1])
            d_loss = self.d_criterion(d_pred, torch.max(d, 1)[1])
            loss = y_loss + d_loss
            loss.backward()
            self.optimizer.step()
        return loss_ave/itr_per_epoch, y_loss_ave/itr_per_epoch
