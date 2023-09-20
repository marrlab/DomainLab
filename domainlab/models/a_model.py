"""
operations that all kinds of models should have
"""

import abc

from torch import nn


class AModel(nn.Module, metaclass=abc.ABCMeta):
    """
    operations that all models (classification, segmentation, seq2seq)
    """
    def cal_loss(self, tensor_x, tensor_y, tensor_d=None, others=None):
        """
        calculate the loss
        """
        list_loss, list_multiplier = self.cal_reg_loss(tensor_x, tensor_y, tensor_d, others)
        loss_reg = self.inner_product(list_loss, list_multiplier)
        return self.cal_task_loss(tensor_x, tensor_y) + loss_reg


    def inner_product(self, list_loss_scalar, list_multiplier):
        """
        compute inner product between list of scalar loss and multiplier
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
    def cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        """
        task independent regularization loss for domain generalization
        """

    def forward(self, tensor_x, tensor_y, tensor_d, others=None):
        """forward.

        :param x:
        :param y:
        :param d:
        """
        return self.cal_loss(tensor_x, tensor_y, tensor_d, others)
