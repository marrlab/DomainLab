"""
globals for the whole package
"""
__docformat__ = "restructuredtext"
import torch


g_inst_component_loss_agg = torch.sum
# component loss refers to aggregation of pixel loss, digit of KL divergences loss
# instance loss currently use torch.sum, which is the same effect as torch.mean, the
# important part is the component aggregation method inside a single instance
