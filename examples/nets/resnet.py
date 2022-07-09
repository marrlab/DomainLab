import torch.nn as nn
from torchvision import models as torchvisionmodels

from libdg.compos.nn import LayerId
from libdg.compos.nn_torchvision import NetTorchVisionBase


class ResNetBase(NetTorchVisionBase):
    """
    """
    def fetch_net(self, flag_pretrain):
        self.net_torchvision = torchvisionmodels.resnet.resnet50(
            pretrained=flag_pretrain)
    # def remove_last_layer(self):


class ResNet4DeepAll(ResNetBase):
    """
    change the size of the last layer
    """
    def __init__(self, flag_pretrain, dim_y):
        super().__init__(flag_pretrain)
        num_final_in = self.net_torchvision.fc.in_features
        self.net_torchvision.fc = nn.Linear(num_final_in, dim_y)


class ResNetNoLastLayer(ResNetBase):
    def __init__(self, flag_pretrain):
        super().__init__(flag_pretrain)
        self.net_torchvision.fc = LayerId()


def build_feat_extract_net(dim_y, remove_last_layer):
    if remove_last_layer:
        return ResNetNoLastLayer(flag_pretrain=True)
    return ResNet4DeepAll(flag_pretrain=True, dim_y=dim_y)
