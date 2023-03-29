"""
Wrapper for MatchDG to conform of model class in DomainLab
"""
from domainlab.models.a_model_classif import AModelClassif


class ModelWrapMatchDGNet(AModelClassif):
    """
    Wrap an arbitrary model with interface:cal_logit_y
    """
    def __init__(self, net, list_str_y, list_str_d=None):
        super().__init__(list_str_y, list_str_d)
        self.check_attribute(net)
        self.net = net

    def check_attribute(self, net):
        """
        not used
        """

    def cal_logit_y(self, tensor_x):
        raise NotImplementedError

    def extract_semantic_feat(self, tensor_x):
        """
        extract semantic feature
        """
        feat = self.net(tensor_x)
        return feat
