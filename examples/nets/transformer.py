"""
make an experiment
"""

from torch import nn
from torchvision.models import vit_b_16
from torchvision.models.feature_extraction import create_feature_extractor

from domainlab.mk_exp import mk_exp
from domainlab.tasks import get_task
from domainlab.models.model_deep_all import mk_deepall


class VIT(nn.Module):
    def __init__(self, num_cls, freeze=True,
                 list_str_last_layer=['getitem_5'],
                 len_last_layer=768):
        super().__init__()
        self.nets = vit_b_16(pretrained=True)
        if freeze:
            # freeze all the network except the final layer
            for param in self.nets.parameters():
                param.requires_grad = False
        self.features_vit_flatten = create_feature_extractor(self.nets,
                                                             return_nodes=list_str_last_layer)
        self.fc = nn.Linear(len_last_layer, num_cls)

    def forward(self, tensor_x):
        """
        compute logits predicts
        """
        x = self.features_vit_flatten(tensor_x)['getitem_5']
        out = self.fc(x)
        return out


def test_transformer():
    """
    test mk experiment API
    """
    # specify domain generalization task
    task = get_task("mini_vlcs")
    # specify backbone to use
    backbone = VIT(num_cls=task.dim_y, freeze=True)
    model = mk_deepall()(backbone)
    # make trainer for model
    exp = mk_exp(task, model, trainer="mldg,dial",
                 test_domain="caltech", batchsize=2, nocu=True)
    exp.execute(num_epochs=2)


if __name__ == '__main__':
    test_transformer()
