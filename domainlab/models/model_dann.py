"""
construct feature extractor, task neural network (e.g. classification) and domain classification
network
"""
from torch.nn import functional as F

from domainlab.compos.nn_zoo.net_adversarial import AutoGradFunReverseMultiply
from domainlab.models.a_model_classif import AModelClassif


def mk_dann(parent_class=AModelClassif):
    """
    Instantiate a Deep Adversarial Net (DAN) model

    Details:
        The model is trained to solve two tasks:
        1. Standard image classification.
        2. Domain classification.
        Here for, a feature extractor is adversarially trained to minimize the loss of the image
        classifier and maximize the loss of the domain classifier.
        For more details, see:
        Ganin, Yaroslav, et al. "Domain-adversarial training of neural networks."
        The journal of machine learning research 17.1 (2016): 2096-2030.

    Args:
        parent_class (AModel, optional): Class object determining the task
        type. Defaults to AModelClassif.

    Returns:
        ModelDAN: model inheriting from parent class

    Input Parameters:
        list_str_y: list of labels,
        list_d_tr: list of training domains
        alpha: total_loss = task_loss + $$\\alpha$$ * domain_classification_loss,
        net_encoder: neural network to extract the features (input: training data),
        net_classifier: neural network (input: output of net_encoder; output: label prediction),
        net_discriminator: neural network (input: output of net_encoder;
        output: prediction of training domain)

    Usage:
        For a concrete example, see:
        https://github.com/marrlab/DomainLab/blob/master/tests/test_mk_exp_dann.py
    """

    class ModelDAN(parent_class):
        """
        anonymous
        """
        def __init__(self, list_str_y, list_d_tr,
                     alpha, net_encoder, net_classifier, net_discriminator):
            """
            See documentation above in mk_dann() function
            """
            super().__init__(list_str_y)
            self.list_d_tr = list_d_tr
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
            return functor_scheduler(trainer=None, alpha=self.alpha)

        def cal_logit_y(self, tensor_x):  # FIXME: this is only for classification
            """
            calculate the logit for softmax classification
            """
            return self.net_classifier(self.net_encoder(tensor_x))

        def _cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others):
            _ = others
            _ = tensor_y
            feat = self.net_encoder(tensor_x)
            logit_d = self.net_discriminator(
                AutoGradFunReverseMultiply.apply(feat, self.alpha))
            _, d_target = tensor_d.max(dim=1)
            lc_d = F.cross_entropy(logit_d, d_target, reduction="none")
            return [lc_d], [self.alpha]
    return ModelDAN
