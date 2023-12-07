"""
operations that all kinds of models should have
"""

import abc

from torch import nn


class AModel(nn.Module, metaclass=abc.ABCMeta):
    """
    operations that all models (classification, segmentation, seq2seq)
    """
    def __init__(self):
        super().__init__()
        self._decoratee = None
        self.list_d_tr = None
        self.visitor = None

    def extend(self, model):
        """
        extend the loss of the decoratee
        """
        self._decoratee = model

    @property
    def metric4msel(self):
        """
        metric for model selection
        """
        raise NotImplementedError

    @property
    def multiplier4task_loss(self):
        """
        the multiplier for task loss is default to 1 except for vae family models
        """
        return 1.0

    def cal_loss(self, tensor_x, tensor_y, tensor_d=None, others=None):
        """
        calculate the loss
        """
        list_loss, list_multiplier = self.cal_reg_loss(tensor_x, tensor_y, tensor_d, others)
        loss_reg = self.inner_product(list_loss, list_multiplier)
        loss_task_alone = self.cal_task_loss(tensor_x, tensor_y)
        loss_task = self.multiplier4task_loss * loss_task_alone
        return loss_task + loss_reg, list_loss, loss_task_alone

    def inner_product(self, list_loss_scalar, list_multiplier):
        """
        compute inner product between list of scalar loss and multiplier
        - the first dimension of the tensor v_reg_loss is mini-batch
        the second dimension is the number of regularizers
        - the vector mmu has dimension the number of regularizers
        """
        list_tuple = zip(list_loss_scalar, list_multiplier)
        rst = [mtuple[0]*mtuple[1] for mtuple in list_tuple]
        return sum(rst)    # FIXME: is "sum" safe to pytorch?

    @abc.abstractmethod
    def cal_task_loss(self, tensor_x, tensor_y):
        """
        Calculate the task loss

        :param tensor_x: input
        :param tensor_y: label
        :return: task loss
        """

    @abc.abstractmethod
    def _cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        """
        task independent regularization loss for domain generalization
        """

    def cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        """
        task independent regularization loss for domain generalization
        """
        loss_reg, mu = self._extend_loss(
            tensor_x, tensor_y, tensor_d, others)
        loss_reg_, mu_ = self._cal_reg_loss(
            tensor_x, tensor_y, tensor_d, others)
        if loss_reg is not None:
            return loss_reg_ + loss_reg, mu_ + mu
        return loss_reg_, mu_

    def _extend_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        """
        combine losses from two models
        """
        if self._decoratee is not None:
            return self._decoratee.cal_reg_loss(
                tensor_x, tensor_y, tensor_d, others)
        return None, None

    def forward(self, tensor_x, tensor_y, tensor_d, others=None):
        """forward.

        :param x:
        :param y:
        :param d:
        """
        return self.cal_loss(tensor_x, tensor_y, tensor_d, others)

    def save(self, suffix=None):
        """
        persist model to disk
        """
        if self.visitor is None:
            return
        self.visitor.save(self, suffix)
        return
    
    def load(self, suffix=None):
        """
        load model from disk
        """
        if self.visitor is None:
            return None
        return self.visitor.load(suffix)
    
    def set_saver(self, visitor):
        self.visitor = visitor