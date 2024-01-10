"""
DomainLab API CODING
"""

import torch
from torch import nn
from torchvision.models import vit_b_16
from torchvision.models.feature_extraction import create_feature_extractor

from domainlab.mk_exp import mk_exp
from domainlab.tasks import get_task
from domainlab.models.model_dann import mk_dann
from domainlab.models.model_jigen import mk_jigen


class VIT(nn.Module):
    """
    Vision transformer as feature extractor
    """
    def __init__(self, freeze=True,
                 list_str_last_layer=['getitem_5'],
                 len_last_layer=768):
        super().__init__()
        self.nets = vit_b_16(pretrained=True)
        if freeze:
            # freeze all the network except the final layer,
            # for fast code execution, set freeze=False
            # in case of enough computation resources
            for param in self.nets.parameters():
                param.requires_grad = False
        self.features_vit_flatten = \
            create_feature_extractor(self.nets,
                                     return_nodes=list_str_last_layer)

    def forward(self, tensor_x):
        """
        compute logits predicts
        """
        out = self.features_vit_flatten(tensor_x)['getitem_5']
        return out


def test_transformer():
    """
    test mk experiment API
    """
    # specify domain generalization task
    task = get_task("mini_vlcs")
    # specify neural network to use as feature extractor
    net_feature = VIT(freeze=True)
    # since the size of feature is 768
    net_classifier = nn.Linear(768, task.dim_y)

    # see documentation for each arguments below
    model_dann = mk_dann()(net_encoder=net_feature,
                           net_classifier=net_classifier,
                           net_discriminator=nn.Linear(768,2),
                           list_str_y=task.list_str_y,
                           list_d_tr=["labelme", "sun"],
                           alpha=1.0)
    # see documentation for each argument below
    model_jigen = mk_jigen()(net_encoder=net_feature,
                             net_classifier_class=net_classifier,
                             net_classifier_permutation=nn.Linear(768, 32),
                             list_str_y=task.list_str_y,
                             coeff_reg=1.0, n_perm=31)

    model_dann.extend(model_jigen) # let Jigen decorate DANN
    model = model_dann
    # make trainer for model, here we decorate trainer mldg with dial
    exp = mk_exp(task, model, trainer="mldg_dial",
                 test_domain="caltech", batchsize=2, nocu=True)
    exp.execute(num_epochs=2)
