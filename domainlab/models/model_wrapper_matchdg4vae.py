"""
Wrapper for MatchDG to conform of model class in DomainLab
"""
from domainlab.models.a_model_classif import AModelClassif


class ModelWrapMatchDGVAE(AModelClassif):
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
        return self.net.cal_logit_y(tensor_x)

    def extract_semantic_feat(self, tensor_x):
        """
        extract semantic feature
        """
        feat = self.net.extract_semantic_features(tensor_x)
        return feat

    def hyper_init(self, functor_scheduler, trainer=None):
        self.net.hyper_init(functor_scheduler, trainer)

    def hyper_update(self, epoch, fun_scheduler):
        self.net.hyper_update(epoch, fun_scheduler)
