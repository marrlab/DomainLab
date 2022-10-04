from domainlab.models.a_model_classif import AModelClassif
from domainlab.utils.override_interface import override_interface
from domainlab.utils.utils_classif import get_label_na, logit2preds_vpic


class ModelDeepAll(AModelClassif):
    def __init__(self, net, list_str_y, list_str_d=None):
        super().__init__(list_str_y, list_str_d)
        self.add_module("net", net)

    @override_interface(AModelClassif)
    def cal_logit_y(self, tensor_x):
        """
        calculate the logit for softmax classification
        """
        logit_y = self.net(tensor_x)
        return logit_y

    def forward(self, tensor_x, tensor_y, tensor_d):
        return self.cal_loss(tensor_x, tensor_y, tensor_d)

    def cal_loss(self, tensor_x, tensor_y, tensor_d):
        lc_y = self.cal_task_loss(tensor_x, tensor_y)
        return lc_y
