import torch.nn as nn
from domainlab.compos.nn_zoo.nn import DenseNet
import torch

def test_DenseNet():
    x = torch.rand(20, 256*6*6)
    model = DenseNet(256*6*6)
    h = model(x)
