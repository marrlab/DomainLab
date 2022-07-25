"""
operations that all claasification model should have
"""

import abc
import torch.nn as nn
from libdg.utils.utils_class import store_args


class AModelClassif(nn.Module, metaclass=abc.ABCMeta):
    """
    operations that all claasification model should have
    """
    match_feat_fun_na = "cal_logit_y"

    @abc.abstractmethod
    def cal_loss(self, *tensors):
        """
        calculate the loss
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cal_logit_y(self, tensor_x):
        """
        calculate the logit for softmax classification
        """
        raise NotImplementedError

    @store_args
    def __init__(self, list_str_y, list_d_tr=None):
        """
        :param list_str_y: list of fixed order, each element is a class label
        """
        super().__init__()

    @abc.abstractmethod
    def infer_y_vpicn(self, tensor):
        """
        :param tensor: input
        :return: vpicn
            v: vector of one-hot class label,
            p: vector of probability,
            i: class label index,
            c: confidence: maximum probability,
            n: list of name of class
        """
        raise NotImplementedError

    @property
    def dim_y(self):
        """
        the class embedding dimension
        """
        return len(self.list_str_y)
