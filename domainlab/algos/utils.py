"""
network builder utils
"""
from torch import nn


def split_net_feat_last(net):
    """
    split a pytorch module into feature extractor and last layer
    """
    net_classifier = list(net.children())[-1]
    net_invar_feat = nn.Sequential(*(list(net.children())[:-1]))
    return net_invar_feat, net_classifier
