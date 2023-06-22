"""
Wrapper for MatchDG to conform of model class in DomainLab
"""
from domainlab.models.a_model_classif import AModelClassif


class ModelWrapMatchDGLogit(AModelClassif):
    """
    Wrap an arbitrary model with interface:cal_logit_y
    """
    def __init__(self, net, list_str_y, list_str_d=None):
        super().__init__(list_str_y, list_str_d)
        self.check_attribute(net)
        self.net = net

    def check_attribute(self, net):
        """
        """
        fun_name = AModelClassif.match_feat_fun_na
        if not hasattr(net, fun_name):
            raise RuntimeError(
                "model to be wrapped must inherit base class ",
                str(AModelClassif.__class__),
                " with attribute:", fun_name)

    def cal_logit_y(self, tensor_x):
        return self.net.cal_logit_y(tensor_x)

    def cal_loss(self, tensor_x, tensor_y, tensor_d=None, others=None):
        return self.net.cal_loss(tensor_x, tensor_y, tensor_d)

    def cal_reg_loss(self, tensor_x, tensor_y, tensor_d=None, others=None):
        return self.net.cal_loss(tensor_x, tensor_y, tensor_d)  # @FIXME: this is wrong

    def forward(self, tensor_x):
        """
        calculate features to be matched
        """
        return self.extract_semantic_feat(tensor_x)

    def extract_semantic_feat(self, tensor_x):
        """
        extract semantic feature
        """
        logit_y = self.net.cal_logit_y(tensor_x)  # @FIXME: match other features instead of logit
        return logit_y
