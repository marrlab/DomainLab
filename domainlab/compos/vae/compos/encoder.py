"""
Pytorch image: i_channel, i_h, i_w
Location-Scale Encoder: SoftPlus
"""
import torch
import torch.distributions as dist
import torch.nn as nn

from domainlab.compos.nn_zoo.net_conv_conv_bn_pool_2 import \
    mk_conv_bn_relu_pool
from domainlab.compos.nn_zoo.nn import DenseNet
from domainlab.compos.utils_conv_get_flat_dim import get_flat_dim


class LSEncoderConvBnReluPool(nn.Module):
    """Location-Scale Encoder with Convolution,
    Batch Normalization, Relu and Pooling.
    Softplus for scale
    """
    def __init__(self, z_dim: int, i_channel, i_h, i_w, conv_stride):
        """
        :param z_dim:
        nn.Sequential allows output dim to be zero.
        So z_dim here can be set to be zero
        :param i_channel:
        :param i_h:
        :param i_w:
        :param conv_stride:
        """
        super().__init__()
        self.i_channel = i_channel
        self.i_h = i_h
        self.i_w = i_w

        self.conv = mk_conv_bn_relu_pool(self.i_channel,
                                         conv_stride=conv_stride)
        # conv-bn-relu-pool-conv-bn-relu-pool(no activation)
        self.flat_dim = get_flat_dim(self.conv, i_channel, i_h, i_w)
        self.fc_loc = nn.Sequential(nn.Linear(self.flat_dim, z_dim))
        self.fc_scale = nn.Sequential(nn.Linear(self.flat_dim, z_dim),
                                      nn.Softplus())  # for scale calculation

        # initialization
        torch.nn.init.xavier_uniform_(self.fc_loc[0].weight)
        self.fc_loc[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc_scale[0].weight)
        self.fc_scale[0].bias.data.zero_()

    def forward(self, img):
        """.
        :param img:
        """
        hidden = self.conv(img)
        hidden = hidden.view(-1, self.flat_dim)
        zd_q_loc = self.fc_loc(hidden)
        zd_q_scale = self.fc_scale(hidden) + 1e-7
        q_zd = dist.Normal(zd_q_loc, zd_q_scale)
        zd_q = q_zd.rsample()  # Reparameterization trick
        return q_zd, zd_q


class LSEncoderDense(nn.Module):
    """
    Location-Scale Encoder with DenseNet as feature extractor
    Softplus for scale
    """
    def __init__(self, z_dim, dim_input, dim_h=4096):
        """
        :param z_dim:
        nn.Sequential allows output dim to be zero.
        So z_dim here can be set to be zero
        :param i_channel:
        :param i_h:
        :param i_w:
        :param conv_stride:
        """
        super().__init__()
        self.net_feat = DenseNet(
            input_flat_size=dim_input, out_hidden_size=dim_h)
        # conv-bn-relu-pool-conv-bn-relu-pool(no activation)
        self.fc_loc = nn.Sequential(nn.Linear(dim_h, z_dim))
        self.fc_scale = nn.Sequential(nn.Linear(dim_h, z_dim),
                                      nn.Softplus())  # for scale calculation

        # initialization
        torch.nn.init.xavier_uniform_(self.fc_loc[0].weight)
        self.fc_loc[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc_scale[0].weight)
        self.fc_scale[0].bias.data.zero_()

    def forward(self, tensor_x):
        """.
        :param tensor_x:
        """
        hidden = self.net_feat(tensor_x)
        zd_q_loc = self.fc_loc(hidden)
        zd_q_scale = self.fc_scale(hidden) + 1e-7
        q_zd = dist.Normal(zd_q_loc, zd_q_scale)
        zd_q = q_zd.rsample()  # Reparameterization trick
        return q_zd, zd_q
