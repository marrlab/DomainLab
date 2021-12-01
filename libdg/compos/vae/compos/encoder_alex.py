import torch
import torch.nn as nn
import torch.distributions as dist
from libdg.compos.nn_alex import AlexNetNoLastLayer



class Encoder4096(nn.Module):
    """
    """
    def __init__(self, z_dim, flag_pretrain):
        """__init__.
        :param hidden_size:
        """
        super().__init__()
        self.alex_no_last = AlexNetNoLastLayer(flag_pretrain)
        self.net_fc_mean = nn.Sequential(nn.Linear(4096, z_dim))   # FIXME
        self.net_fc_scale = nn.Sequential(nn.Linear(4096, z_dim), nn.Softplus())  # for scale calculation

        torch.nn.init.xavier_uniform_(self.net_fc_mean[0].weight)
        self.net_fc_mean[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.net_fc_scale[0].weight)
        self.net_fc_scale[0].bias.data.zero_()

    def forward(self, x):
        """
        :param x:
        """
        feature = self.alex_no_last(x)
        zd_q_loc = self.net_fc_mean(feature)
        zd_q_scale = self.net_fc_scale(feature) + 1e-7
        q_zd = dist.Normal(zd_q_loc, zd_q_scale)
        zd_q = q_zd.rsample()  # Reparameterization trick
        return q_zd, zd_q
