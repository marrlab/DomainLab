import abc

import torch

from domainlab.models.a_model_classif import AModelClassif
from domainlab.utils.utils_classif import get_label_na, logit2preds_vpic


class AModelCustom(AModelClassif):
    """AModelCustom."""
    @abc.abstractmethod
    def dict_net_module_na2arg_na(self):
        """dict_net_module_na2arg_na.
        A dictionary with the key being the pytorch module name and value
        being the commandline argument name
        """
