"""
In PyTorch, images are represented as [channels, height, width]
"""
import torch
import torch.nn as nn

from domainlab.compos.nn_zoo.nn import DenseNet
from domainlab.compos.utils_conv_get_flat_dim import get_flat_dim


def mk_conv_bn_relu_pool(i_channel, conv_stride=1, max_pool_stride=2):
    """
    Convolution, batch norm, maxpool_2d
    Convolution with maxpool_2d as last operation
    :param i_channel:
    :param conv_stride:
    :param max_pool_stride:
    """
    conv_net = nn.Sequential(
        nn.Conv2d(i_channel, 32, kernel_size=5,
                  stride=conv_stride, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=max_pool_stride),
        nn.Conv2d(32, 64, kernel_size=5, stride=conv_stride, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=max_pool_stride),
    )
    torch.nn.init.xavier_uniform_(conv_net[0].weight)
    torch.nn.init.xavier_uniform_(conv_net[4].weight)
    return conv_net


class NetConvBnReluPool2L(nn.Module):
    def __init__(self, i_c, i_h, i_w, conv_stride, dim_out_h):
        """
        :param dim_out_h:
        """
        super().__init__()
        self.conv_net = mk_conv_bn_relu_pool(i_c, conv_stride)
        ###
        self.hdim = get_flat_dim(self.conv_net, i_c, i_h, i_w)
        self.layer_last = nn.Linear(self.hdim, dim_out_h)

    def forward(self, tensor_x):
        """
        :param tensor_x: image
        """
        conv_out = self.conv_net(tensor_x)  # conv-bn-relu-pool-conv-bn-relu-pool(no activation)
        flat = conv_out.view(-1, self.hdim)   # 1024 =   64 * (4*4)
        hidden = self.layer_last(flat)
        return hidden


class NetConvDense(nn.Module):
    """
    - For direct topic inference
    - For custom deep_all, which is extracting the path of VAE from encoder
      until classifier. note in encoder, there is extra layer of hidden to mean
      and scale, in this component, it is replaced with another hidden layer.
    """
    def __init__(self, i_c, i_h, i_w, conv_stride, dim_out_h, args, dense_layer=None):
        """
        :param dim_out_h:
        """
        super().__init__()
        self.conv_net = mk_conv_bn_relu_pool(i_c, conv_stride)
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
