"""
use random start to generate adversarial images
"""
import torch
from torch.autograd import Variable
import torch.nn as nn

try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from domainlab.algos.trainers.train_basic import TrainerBasic


class TrainerFishr(TrainerBasic):
    """
    Trainer Domain Invariant Adversarial Learning
    """
    def __init__(self):
        super().__init__()
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))

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
        logits = self.model.cal_logit_y(tensor_x)
        loss_erm = self.bce_extended(logits, vec_y).sum()
        with backpack(BatchGrad()):
            loss_erm.backward(
                inputs=list(self.model.parameters(), retain_graph=True, create_graph=True)
            )
        dict_grads = OrderedDict(
            [name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1)
             for name, weights in self.model.named_parameters()
            ]
        )
        var = self.cal_variance(dict_grads)
        return loss_fishr

    def cal_variance(dict_grads):
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_mini
