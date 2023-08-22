"""
use random start to generate adversarial images
"""
import torch
import copy
from torch.autograd import Variable
import torch.nn as nn
from collections import OrderedDict

try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad, Variance
except:
    backpack = None

from domainlab.algos.trainers.train_basic import TrainerBasic

_bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))

class TrainerFishr(TrainerBasic):
    """
    Trainer Domain Invariant Adversarial Learning
    """
    def tr_epoch(self, epoch):
        self.model.train()
        self.model.convert4backpack()
        self.epo_loss_tr = 0
        for ind_batch, (tensor_x, vec_y, vec_d, *_) in enumerate(self.loader_tr):
            tensor_x, vec_y, vec_d = \
                tensor_x.to(self.device), vec_y.to(self.device), vec_d.to(self.device)
            self.optimizer.zero_grad()
            loss_erm = self.model.cal_loss(tensor_x, vec_y, vec_d)  # @FIXME
            loss_fishr = self.cal_fishr(tensor_x, vec_y, vec_d)
            loss = loss_erm.sum() + self.aconf.gamma_reg * loss_fishr
            loss.backward()
            self.optimizer.step()
            self.epo_loss_tr += loss.detach().item()
            self.after_batch(epoch, ind_batch)
        flag_stop = self.observer.update(epoch)  # notify observer
        return flag_stop

    def cal_fishr(self, tensor_x, vec_y, vec_d):
        """
        use backpack
        """
        loss = self.model.cal_task_loss(tensor_x.clone(), vec_y)

        with backpack(Variance()):
            loss.backward(
                inputs=list(self.model.parameters()), retain_graph=True, create_graph=True
            )

        for name, param in self.model.named_parameters():
            print(name)
            print(".grad.shape:             ", param.variance.shape)

        dict_variance = OrderedDict(
            [(name, weights.variance.clone())
             for name, weights in self.model.named_parameters()
             ])
        return 0
