'''
Code coverage issues:
    https://app.codecov.io/gh/marrlab/DomainLab/blob/master/domainlab/compos/nn_zoo/net_conv_conv_bn_pool_2.py
    - lines 66-67, 69-71, 73
    - lines 79-82
'''

import torch
from domainlab.compos.nn_zoo.nn import DenseNet
from domainlab.compos.nn_zoo.net_conv_conv_bn_pool_2 import NetConvDense


def test_netconvdense1():
    inpu = torch.randn(1, 3, 28, 28)
    model = NetConvDense(i_c=3, i_h=28, i_w=28, conv_stride=1, dim_out_h=32, args=None, dense_layer=None)
    model(inpu)
    

def test_netconvdense2():
    inpu = torch.randn(1,3,28,28)
    dense_layers = DenseNet(1024, out_hidden_size=32)
    model = NetConvDense(i_c=3, i_h=28, i_w=28, conv_stride=1, dim_out_h=32, args=None, dense_layer=dense_layers)
    model(inpu)
