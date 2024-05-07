"""
use random start to generate adversarial images
"""
import torch
from torch import nn
from torch.nn import functional as F
import torch.autograd as autograd
from domainlab.algos.trainers.train_basic import TrainerBasic


class TrainerIRM(TrainerBasic):
    """
    The goal is to minimize the variance of the domain-level variance of the gradients.
    This aligns the domain-level loss landscapes locally around the final weights, reducing
    inconsistencies across domains.

    For more details, see: Alexandre Ramé, Corentin Dancette, and Matthieu Cord.
        "Fishr: Invariant gradient variances for out-of-distribution generalization."
        International Conference on Machine Learning. PMLR, 2022.
    """
    def _cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        """
        Let trainer behave like a model, so that other trainer could use it
        """
        _ = tensor_d
        _ = others
        y = tensor_y
        logits = self.model.cal_logit_y(tensor_x)
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        loss_irm = torch.sum(grad_1 * grad_2)
        return [loss_irm], [self.aconf.gamma_reg]
