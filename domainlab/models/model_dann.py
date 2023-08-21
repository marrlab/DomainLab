"""
construct feature extractor, task neural network (e.g. classification) and domain classification network
"""
from torch.nn import functional as F

from domainlab.compos.nn_zoo.net_adversarial import AutoGradFunReverseMultiply
from domainlab.models.a_model_classif import AModelClassif


def mk_dann(parent_class=AModelClassif):
    """
    Instantiate a Deep Adversarial Net (DAN) model

    Args:
        parent_class (AModel, optional): Class object determining the task
        type. Defaults to AModelClassif.

    Returns:
        ModelDAN: model inheriting from parent class

    Notes:
        list_str_y: list of labels
        list_str_d: list of domains
        alpha: total_loss = task_loss + \alpha * regularization_loss
        net_encoder: neural network to extract the features, input: training data
        net_classifier: neural network, input: output of net_encoder, output: label
        net_discriminator: neural network, input: output of net_encoder, output: training domain
    """
    class ModelDAN(parent_class):
        """
        anonymous
        """
        def __init__(self, list_str_y, list_str_d,
                     alpha, net_encoder, net_classifier, net_discriminator):
            """
            See documentation above in mk_dann() function
            """             
            super().__init__(list_str_y, list_str_d)
            self.alpha = alpha
            self.net_encoder = net_encoder
            self.net_classifier = net_classifier
            self.net_discriminator = net_discriminator

        def hyper_update(self, epoch, fun_scheduler):
            """hyper_update.
            :param epoch:
            :param fun_scheduler: the hyperparameter scheduler object
            """
            dict_rst = fun_scheduler(epoch)  # the __call__ method of hyperparameter scheduler
            self.alpha = dict_rst["alpha"]

        def hyper_init(self, functor_scheduler):
            """hyper_init.
            :param functor_scheduler:
            """
            return functor_scheduler(alpha=self.alpha)

        def cal_logit_y(self, tensor_x):  # FIXME: this is only for classification
            """
            calculate the logit for softmax classification
            """
            return self.net_classifier(self.net_encoder(tensor_x))

        def cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
            feat = self.net_encoder(tensor_x)
            logit_d = self.net_discriminator(
                AutoGradFunReverseMultiply.apply(feat, self.alpha))
            _, d_target = tensor_d.max(dim=1)
            lc_d = F.cross_entropy(logit_d, d_target, reduction="none")
            return self.alpha*lc_d
    return ModelDAN
