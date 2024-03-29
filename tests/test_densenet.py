"""
Code coverage issues:
    https://app.codecov.io/gh/marrlab/DomainLab/blob/master/domainlab/compos/nn_zoo/net_conv_conv_bn_pool_2.py
    - lines 66-67, 69-71, 73
    - lines 79-82
"""
import gc

import torch

from domainlab.compos.nn_zoo.net_conv_conv_bn_pool_2 import NetConvDense
from domainlab.compos.nn_zoo.nn import DenseNet


def test_netconvdense1():
    """
    test convdensenet
    """
    inpu = torch.randn(1, 3, 28, 28)
    model = NetConvDense(
        isize=(3, 28, 28), conv_stride=1, dim_out_h=32, args=None, dense_layer=None
    )
    model(inpu)
    del model
    torch.cuda.empty_cache()
    gc.collect()


def test_netconvdense2():
    """
    test convdensenet
    """
    inpu = torch.randn(1, 3, 28, 28)
    dense_layers = DenseNet(1024, out_hidden_size=32)
    model = NetConvDense(
        isize=(3, 28, 28),
        conv_stride=1,
        dim_out_h=32,
        args=None,
        dense_layer=dense_layers,
    )
    model(inpu)
    del model
    torch.cuda.empty_cache()
    gc.collect()
