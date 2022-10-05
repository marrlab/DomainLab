import torch.nn as nn
from torch.distributions import Dirichlet


class EncoderH2Dirichlet(nn.Module):
    """
    hidden representation to Dirichlet Distribution
    """

    def __init__(self, dim_h, dim_topic, device):
        """
        """
        super().__init__()
        self.net_fc = nn.Linear(dim_h, dim_topic)
        self.layer_bn = nn.BatchNorm1d(dim_topic)
        self.device = device

    def forward(self, hidden):
        """
        :param hidden:
        """
        alphas_batch = self.layer_bn(self.net_fc(hidden)).exp()
        q_topic = Dirichlet(alphas_batch)
        topic_q = q_topic.rsample().to(self.device)
        return q_topic, topic_q
