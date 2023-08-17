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
        extend_model = extend(copy.deepcopy(self.model))
        logits = extend_model.cal_logit_y(tensor_x.clone()).clone()
        # logits = extend_model(tensor_x.clone()).clone()
        loss_erm = _bce_extended(logits, vec_y).sum()
        #with backpack(BatchGrad()):
        #    loss_erm.backward(
        #        inputs=list(self.model.parameters()), retain_graph=True, create_graph=True)


        #dict_grads = OrderedDict(
        #    [name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1)
        #     for name, weights in self.model.named_parameters()
        #     ]
        #)

        #  backpack should be able to compute the variance directly
        with backpack(Variance()):
            loss_erm.backward(
                inputs=list(self.model.parameters()), retain_graph=True, create_graph=True
            )
        breakpoint()

        dict_variance = OrderedDict(
            [(name, weights.variance.clone().view(weights.variance.size(0), -1))
             for name, weights in self.model.named_parameters()
            ]
        )

        return 0
