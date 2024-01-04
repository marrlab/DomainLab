import torch
import torch.nn as nn
from torch.distributions import Dirichlet


class EncoderH2Dirichlet(nn.Module):
    """
    hidden representation to Dirichlet Distribution
    """

    def __init__(self, dim_topic, device):
        """
        """
        super().__init__()
        self.layer_bn = nn.BatchNorm1d(dim_topic)
        self.layer_concentration = nn.Softplus()
        self.device = device

    def forward(self, hidden):
        """
        :param hidden:
        """
        feat_bnorm = self.layer_bn(hidden)
        # alphas_batch = torch.log(1 + feat_bnorm.exp())
        alphas_batch = self.layer_concentration(feat_bnorm)
        q_topic = Dirichlet(alphas_batch + 1e-6)
        topic_q = q_topic.rsample().to(self.device)
        return q_topic, topic_q
