"""
operations that all claasification model should have
"""

import abc
import torch
import torch.nn as nn
from torch.nn import functional as F
from domainlab.utils.utils_class import store_args
from domainlab.utils.utils_classif import logit2preds_vpic, get_label_na


class AModelClassif(nn.Module, metaclass=abc.ABCMeta):
    """
    operations that all classification model should have
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
        with torch.no_grad():
            logit_y = self.cal_logit_y(tensor)
        vec_one_hot, prob, ind, confidence = logit2preds_vpic(logit_y)
        na_class = get_label_na(ind, self.list_str_y)
        return vec_one_hot, prob, ind, confidence, na_class

    @property
    def dim_y(self):
        """
        the class embedding dimension
        """
        return len(self.list_str_y)
