"""
operations that all kinds of models should have
"""

import abc

from torch import nn


class AModel(nn.Module, metaclass=abc.ABCMeta):
    """
    operations that all models (classification, segmentation, seq2seq should have)
    """

    @abc.abstractmethod
    def cal_loss(self, *tensors):
        """
        calculate the loss
        """
        raise NotImplementedError

    def cal_task_loss(self, tensor_x, tensor_y):
        """
        Calculate the task loss

        :param tensor_x: input
        :param tensor_y: label
        :return: task loss
        """
        raise NotImplementedError
