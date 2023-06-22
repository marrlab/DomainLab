import torch
import torch.distributions as dist
import torch.nn as nn


class LSCondPriorLinearBnReluLinearSoftPlus(nn.Module):
    """
    Location-Scale: from hyper-prior to current layer prior distribution
    """
    def __init__(self, hyper_prior_dim, z_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            self.hidden_dim = z_dim
        self.net_linear_bn_relu = nn.Sequential(
            nn.Linear(hyper_prior_dim, self.hidden_dim, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU())
        self.fc_loc = nn.Sequential(nn.Linear(self.hidden_dim, z_dim))
        # No activation, because latent code z variable can take both negative and positive value
        self.fc_scale = nn.Sequential(nn.Linear(self.hidden_dim, z_dim), nn.Softplus())

        # initialization
        torch.nn.init.xavier_uniform_(self.net_linear_bn_relu[0].weight)
        torch.nn.init.xavier_uniform_(self.fc_loc[0].weight)
        self.fc_loc[0].bias.data.zero_()   # No Bias
        torch.nn.init.xavier_uniform_(self.fc_scale[0].weight)
        self.fc_scale[0].bias.data.zero_()   # No Bias

    def forward(self, hyper_prior):
        """
        from hyper-prior value to current latent variable distribution
        """
        hidden = self.net_linear_bn_relu(hyper_prior)
        z_loc = self.fc_loc(hidden)
        z_scale = self.fc_scale(hidden) + 1e-7
        prior = dist.Normal(z_loc, z_scale)
        return prior
