import abc
import torch
from domainlab.models.a_model_classif import AModelClassif
from domainlab.compos.zoo_nn import FeatExtractNNBuilderChainNodeGetter
from domainlab.utils.utils_classif import logit2preds_vpic, get_label_na


class AModelCustom(AModelClassif):
    """AModelCustom."""

    def __init__(self, list_str_y, list_str_d=None):
        """__init__.
        :param net:
        :param list_str_y:
        :param list_str_d:
        """
        super().__init__(list_str_y, list_str_d)

    @property
    @abc.abstractmethod
    def dict_net_module_na2arg_na(self):
        """dict_net_module_na2arg_na.
        A dictionary with the key being the pytorch module name and value
        being the commandline argument name
        """
        raise NotImplementedError

    def cal_logit_y(self, tensor_x):
        """
        calculate the logit for softmax classification
        """
        raise NotImplementedError

    def forward(self, tensor_x, tensor_y, tensor_d):
        """forward.

        :param tensor_x:
        :param tensor_y:
        :param tensor_d:
        """
        raise NotImplementedError

    def cal_loss(self, tensor_x, tensor_y, tensor_d):
        """cal_loss.

        :param tensor_x:
        :param tensor_y:
        :param tensor_d:
        """
        raise NotImplementedError

    def infer_y_vpicn(self, tensor):
        """
        :param tensor: input
        :return:
            - v - vector of one-hot class label
            - p - vector of probability
            - i - class label index
            - c - confidence: maximum probability
            - n - list of name of class
        """
        with torch.no_grad():
            logit_y = self.cal_logit_y(tensor)
            vec_one_hot, prob, ind, confidence = logit2preds_vpic(logit_y)
            na_class = get_label_na(ind, self.list_str_y)
            return vec_one_hot, prob, ind, confidence, na_class
