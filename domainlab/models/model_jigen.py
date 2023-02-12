"""
Jigen Model Similar to DANN model
"""
from torch.nn import functional as F

from domainlab.models.a_model_classif import AModelClassif
from domainlab.models.model_dann import mk_dann


def mk_jigen(parent_class=AModelClassif):
    """Instantiate a JiGen model

    Args:
        parent_class (AModel, optional): Class object determining the task
        type. Defaults to AModelClassif.

    Returns:
        ModelJiGen: model inheriting from parent class
    """
    class_dann = mk_dann(parent_class)

    class ModelJiGen(class_dann):
        """
        Jigen Model Similar to DANN model
        """
        def __init__(self, list_str_y, list_str_d,
                     net_encoder,
                     net_classifier_class,
                     net_classifier_permutation,
                     coeff_reg):
            super().__init__(list_str_y, list_str_d,
                             alpha=coeff_reg,
                             net_encoder=net_encoder,
                             net_classifier=net_classifier_class,
                             net_discriminator=net_classifier_permutation)
            self.net_encoder = net_encoder
            self.net_classifier_class = net_classifier_class
            self.net_classifier_permutation = net_classifier_permutation

        def cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
            """
            JiGen don't need domain label but a pre-defined permutation index
            to calculate regularization loss
            """
            vec_perm_ind = tensor_d
            feat = self.net_encoder(tensor_x)
            logit_d = self.net_classifier_permutation(feat)
            # _, target_perm_ind_scalar = vec_perm_ind.max(dim=1)
            target_perm_ind_scalar = vec_perm_ind
            loss_perm = F.cross_entropy(logit_d, target_perm_ind_scalar, reduction="none")
            return loss_perm
    return ModelJiGen
