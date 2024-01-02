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
        self.device = device

    def forward(self, hidden):
        """
        :param hidden:
        """
        # feat_bnorm = self.layer_bn(hidden)
        feat_bnorm = hidden
        alphas_batch = torch.log(1 + feat_bnorm.exp())
        q_topic = Dirichlet(alphas_batch)
        topic_q = q_topic.rsample().to(self.device)
        return q_topic, topic_q
