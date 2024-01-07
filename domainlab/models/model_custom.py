import abc

import torch

from domainlab.models.a_model_classif import AModelClassif
from domainlab.utils.utils_classif import get_label_na, logit2preds_vpic


class AModelCustom(AModelClassif):
    """AModelCustom."""

    def __init__(self, list_str_y, list_str_d=None):
        """__init__.
        :param net:
        :param list_str_y:
        :param list_str_d:
        """
        super().__init__(list_str_y)

    @property
    @abc.abstractmethod
    def dict_net_module_na2arg_na(self):
        """dict_net_module_na2arg_na.
        A dictionary with the key being the pytorch module name and value
        being the commandline argument name
        """
        raise NotImplementedError

    def extract_semantic_feat(self, tensor_x):
        """
        calculate the logit for softmax classification
        """
        raise NotImplementedError
