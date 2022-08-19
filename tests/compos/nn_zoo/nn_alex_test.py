import torch
import torch.nn as nn
from torchvision import models as torchvisionmodels

from domainlab.compos.nn_zoo.nn import LayerId
from domainlab.compos.nn_zoo.nn_torchvision import NetTorchVisionBase
from domainlab.compos.nn_zoo.nn_alex import AlexNetNoLastLayer

def test_AlexNetConvClassif():
    model = AlexNetNoLastLayer(True)
    x = torch.rand(20, 3, 224, 224)
    x = torch.clamp(x, 0, 1)
    x.require_grad = False
    torch.all(x > 0)
    res = model(x)
    res.shape
