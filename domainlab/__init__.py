"""
globals for the whole package
"""
__docformat__ = "restructuredtext"
import torch

g_inst_component_loss_agg = torch.sum
g_tensor_batch_agg = torch.sum
g_list_loss_agg = sum

g_name_num_shared_param_samples_rand_search = "num_shared_param_samples"

def g_list_model_penalized_reg_agg(list_penalized_reg):
    """
    aggregate along the list, but do not diminish the batch structure of the tensor
    """
    return torch.stack(list_penalized_reg, dim=0).sum(dim=0)


g_str_cross_entropy_agg = "none"
# component loss refers to aggregation of pixel loss, digit of KL divergences loss
# instance loss currently use torch.sum, which is the same effect as torch.mean, the
# important part is the component aggregation method inside a single instance
