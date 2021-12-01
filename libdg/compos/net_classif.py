"Classifier"
import torch
import torch.nn as nn
from torch.nn import functional as F


class ClassifDropoutReluLinear(nn.Module):
    """first apply dropout, then relu, then linearly fully connected, without activation"""
    def __init__(self, z_dim, target_dim):
        """
        :param z_dim:
        :param target_dim:
        """
        super().__init__()
        self.op_drop = nn.Dropout()
        self.op_linear = nn.Linear(z_dim, target_dim)
        torch.nn.init.xavier_uniform_(self.op_linear.weight)
        self.op_linear.bias.data.zero_()

    def forward(self, z_vec):
        """
        :param z_vec:
        """
        hidden = F.relu(self.op_drop(z_vec))
        logit = self.op_linear(hidden)
        return logit
