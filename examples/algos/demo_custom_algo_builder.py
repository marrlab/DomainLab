import torch
import torch.nn as nn
from torch.nn import functional as F

from domainlab.models.a_model_classif import AModelClassif
from domainlab.utils.utils_classif import logit2preds_vpic, get_label_na
from domainlab.algos.builder_make import make_algo


class ModelCustom(AModelClassif):    # The model you implemented must start with prefix "Model" and inherit AModelClassif
    """ModelCustom."""

    def __init__(self, list_str_y, list_str_d=None):
        """__init__.
        :param net:
        :param list_str_y:
        :param list_str_d:
        """
        super().__init__(list_str_y, list_str_d)

    def set_nets_from_dictionary(self):
        """set_nets_from_dictionary.
        :param dict_net:
        """
        for i, (key, val) in enumerate(self.dict_net):
            self.add_module("net%d", val)

    @property
    def dict_net(self):
        return {"net": "nname"}

    @property
    def net_predict(self):
        return self.net0

    def cal_logit_y(self, tensor_x):
        """
        calculate the logit for softmax classification
        """
        logit_y = self.net_predict(tensor_x)
        return logit_y

    def infer_y_vpicn(self, tensor):
        """infer_y_vpicn.
        :param tensor:
        """
        with torch.no_grad():
            logit_y = self.net_predict(tensor)
        vec_one_hot, prob, ind, confidence = logit2preds_vpic(logit_y)
        na_class = get_label_na(ind, self.list_str_y)
        return vec_one_hot, prob, ind, confidence, na_class

    def forward(self, tensor_x, tensor_y, tensor_d):
        """forward.

        :param tensor_x:
        :param tensor_y:
        :param tensor_d:
        """
        return self.cal_loss(tensor_x, tensor_y, tensor_d)

    def cal_loss(self, tensor_x, tensor_y, tensor_d):
        """cal_loss.

        :param tensor_x:
        :param tensor_y:
        :param tensor_d:
        """
        logit_y = self.net_predict(tensor_x)
        if (tensor_y.shape[-1] == 1) | (len(tensor_y.shape) == 1):
            y_target = tensor_y
        else:
            _, y_target = tensor_y.max(dim=1)
        lc_y = F.cross_entropy(logit_y, y_target, reduction="none")
        return lc_y


def get_node_na():
    """In your custom python file, this function has to be implemented
    to return the custom algorithm builder"""
    return make_algo(ModelCustom)
