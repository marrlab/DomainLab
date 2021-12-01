"""
In PyTorch, images are represented as [channels, height, width]
"""
import torch
import torch.nn as nn
from libdg.compos.nn import DenseNet


def mk_conv_bn_relu_pool(i_channel, conv_stride=1, max_pool_stride=2):
    """
    Convolution, batch norm, maxpool_2d
    Convolution with maxpool_2d as last operation
    :param i_channel:
    :param conv_stride:
    :param max_pool_stride:
    """
    conv_net = nn.Sequential(
        nn.Conv2d(i_channel, 32, kernel_size=5, stride=conv_stride, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=max_pool_stride),
        nn.Conv2d(32, 64, kernel_size=5, stride=conv_stride, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=max_pool_stride),
    )
    return conv_net


def test_conv_net():
    """test"""
    model = mk_conv_bn_relu_pool(3)
    model


def get_flat_dim(module, i_channel, i_h, i_w, batchsize=5):
    """flat the convolution layer output and get the flat dimension for fully connected network
    :param module:
    :param i_channel:
    :param i_h:
    :param i_w:
    :param batchsize:
    """
    img = torch.randn(i_channel, i_h, i_w)
    img3 = img.repeat(batchsize, 1, 1, 1)  # create batchsize repitition
    conv_output = module(img3)
    flat_dim = conv_output.shape[1] * conv_output.shape[2] * conv_output.shape[3]
    return flat_dim


def test_get_flat_dim():
    model = mk_conv_bn_relu_pool(3)
    get_flat_dim(model, 3, 28, 28)


class NetConvDense(nn.Module):
    """
    1. For direct topic inference
    2. For custom deep_all, which is extracting the path of VAE from encoder
    till classifier. note in encoder, there is extra layer of hidden to mean
    and scale, in this component, it is replaced with another hidden layer.
    """
    def __init__(self, i_c, i_h, i_w, conv_stride, dim_out_h, dense_layer=None):
        """
        :param dim_out_h:
        """
        super().__init__()
        ###
        self.conv_net = mk_conv_bn_relu_pool(i_c, conv_stride)
        torch.nn.init.xavier_uniform_(self.conv_net[0].weight)
        torch.nn.init.xavier_uniform_(self.conv_net[4].weight)
        ###
        self.hdim = get_flat_dim(self.conv_net, i_c, i_h, i_w)
        if dense_layer is None:
            self.dense_layers = DenseNet(self.hdim, out_hidden_size=dim_out_h)
        else:
            self.dense_layers = dense_layer

    def forward(self, tensor_x):
        """
        :param tensor_x: image
        """
        conv_out = self.conv_net(tensor_x)  # conv-bn-relu-pool-conv-bn-relu-pool(no activation)
        flat = conv_out.view(-1, self.hdim)   # 1024 =   64 * (4*4)
        hidden = self.dense_layers(flat)
        return hidden
