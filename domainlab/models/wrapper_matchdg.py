import torch

from domainlab.models.a_model_classif import AModelClassif
from domainlab.utils.utils_classif import logit2preds_vpic, get_label_na


class ModelWrapMatchDGLogit(AModelClassif):
    """
    Wrap an arbitrary model with interface:cal_logit_y
    """
    def __init__(self, net, list_str_y, list_str_d=None):
        super().__init__(list_str_y, list_str_d)
        fun_name = AModelClassif.match_feat_fun_na
        if not hasattr(net, fun_name):
            raise RuntimeError(
                "model to be wrapped must inherit base class ",
                str(AModelClassif.__class__),
                " with attribute:", fun_name)
        self.net = net

    def cal_logit_y(self, tensor_x):
        return self.net.cal_logit_y(tensor_x)

    def cal_loss(self, x, y, d=None):
        return self.net.cal_loss(x, y, d)

    def forward(self, tensor_x):
        """
        calculate features to be matched
        """
        logit_y = self.net.cal_logit_y(tensor_x)  # FIXME: match other features instead of logit
        return logit_y
