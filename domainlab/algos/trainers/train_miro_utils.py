"""
Laplace approximation for Mutual Information estimation
"""
import torch
import torch.nn.functional as F
from torch import nn


class MeanEncoder(nn.Module):
    """Identity function"""
    def __init__(self, inter_layer_feat_shape):
        super().__init__()
        self.inter_layer_feat_shape = inter_layer_feat_shape

    def forward(self, x):
        return x


class VarianceEncoder(nn.Module):
    """Bias-only model with diagonal covariance"""
    def __init__(self, inter_layer_feat_shape, init=0.1, channelwise=True, eps=1e-5):
        super().__init__()
        self.inter_layer_feat_shape = inter_layer_feat_shape
        self.eps = eps

        init = (torch.as_tensor(init - eps).exp() - 1.0).log()
        b_shape = inter_layer_feat_shape
        # FIXME:  the neural network should be responsible for
        # returning the shape
        if channelwise:
            if len(inter_layer_feat_shape) == 4:
                # [B, C, H, W]
                b_shape = (1, inter_layer_feat_shape[1], 1, 1)
            elif len(inter_layer_feat_shape) == 3:
                # CLIP-ViT: [H*W+1, B, C]
                b_shape = (1, 1, inter_layer_feat_shape[2])
            else:
                raise ValueError()

        self.b = nn.Parameter(torch.full(b_shape, init))

    def forward(self, feat_layer_tensor_batch):
        """
        train batch(population) level variance
        """
        return F.softplus(self.b) + self.eps
