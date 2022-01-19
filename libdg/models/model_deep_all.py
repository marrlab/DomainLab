import torch
import torch.nn as nn
from torch.nn import functional as F

from libdg.models.a_model_classif import AModelClassif
from libdg.utils.utils_classif import logit2preds_vpic, get_label_na


class ModelDeepAll(AModelClassif):
    def __init__(self, net, list_str_y, list_str_d=None):
        super().__init__(list_str_y, list_str_d)
        self.net = net

    def cal_logit_y(self, tensor_x):
        """
        calculate the logit for softmax classification
        """
        logit_y = self.net(tensor_x)
        return logit_y

    def infer_y_vpicn(self, tensor):
        with torch.no_grad():
            logit_y = self.net(tensor)
        vec_one_hot, prob, ind, confidence = logit2preds_vpic(logit_y)
        na_class = get_label_na(ind, self.list_str_y)
        return vec_one_hot, prob, ind, confidence, na_class

    def forward(self, tensor_x, tensor_y, tensor_d):
        return self.cal_loss(tensor_x, tensor_y, tensor_d)

    def cal_loss(self, tensor_x, tensor_y, tensor_d):
        logit_y = self.net(tensor_x)
        _, y_target = tensor_y.max(dim=1)
        lc_y = F.cross_entropy(logit_y, y_target, reduction="none")
        return lc_y
