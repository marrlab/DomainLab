"""
make an experiment
"""

from torch import nn
from torchvision.models import vit_b_16
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

from domainlab.mk_exp import mk_exp
from domainlab.tasks import get_task
from domainlab.models.model_deep_all import mk_deepall

class FCLayer(nn.Module):
    def __init__(self, model, num_classes):
        super(FCLayer, self).__init__()
        self.model = model
        num_ftrs = 768
        self.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)['getitem_5']
        out = self.fc(x)
        return out


class PTVIT(nn.Module):
    def __init__(self, num_cls=15, freeze=True):
        super().__init__()
        self.model = vit_b_16(pretrained=True)
        if freeze:
        # freeze all the network except the final layer
            for param in self.model.parameters():
                param.requires_grad = False
        # These two lines not needed but,
        # you would use them to work out which node you want
        nodes, eval_nodes = get_graph_node_names(self.model)
        #print('model nodes:', nodes)
        self.features_encoder = create_feature_extractor(self.model, return_nodes=['encoder'])
        #print('features_encoder:', self.features_encoder)
        self.features_vit_flatten = create_feature_extractor(self.model, return_nodes=['getitem_5'])
        self.features_fc = create_feature_extractor(self.model, return_nodes=['heads'])
        self.model_final = FCLayer(self.features_vit_flatten, num_cls)

    def forward(self, tensor_x):
        out = self.model_final(tensor_x)
        return out

def test_mk_exp():
    """
    test mk experiment API
    """
    # specify domain generalization task
    task = get_task("mini_vlcs")
    # specify backbone to use
    backbone = PTVIT(num_cls=task.dim_y, freeze=True)
    model = mk_deepall()(backbone)
    # make trainer for model
    exp = mk_exp(task, model, trainer="mldg,dial", test_domain="caltech", batchsize=2)
    exp.execute(num_epochs=2)

if __name__ == '__main__':
    test_mk_exp()
