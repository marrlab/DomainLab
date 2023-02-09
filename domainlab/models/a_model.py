"""
operations that all kinds of models should have
"""

import abc

from torch import nn


class AModel(nn.Module, metaclass=abc.ABCMeta):
    """
    operations that all models (classification, segmentation, seq2seq)
    """

    @abc.abstractmethod
    def cal_loss(self, *tensors):
        """
        calculate the loss
        """

    @abc.abstractmethod
    def cal_task_loss(self, tensor_x, tensor_y):
        """
        Calculate the task loss

        :param tensor_x: input
        :param tensor_y: label
        :return: task loss
        """
    @abc.abstractmethod
    def cal_reg_loss(self, tensor_x, tensor_y, tensor_d):
        """
        task independent regularization loss for domain generalization
        """
