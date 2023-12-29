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
        feat_fc = self.net_fc(hidden)
        feat_bnorm = self.layer_bn(feat_fc)
        alphas_batch = feat_bnorm.exp()
        q_topic = Dirichlet(alphas_batch)
        topic_q = q_topic.rsample().to(self.device)
        return q_topic, topic_q
