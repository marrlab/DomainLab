"""
Template class to inherit for custom model
"""
import torch
from torch.nn import functional as F

from domainlab.algos.builder_custom import make_basic_trainer
from domainlab.models.model_custom import AModelCustom


class ModelCustom(AModelCustom):
    """
    Template class to inherit from if user need custom neural network
    """
    @property
    def dict_net_module_na2arg_na(self):
        """
        we use this property to associate the module "net_aux" with commandline argument
        "my_custom_arg_name", so that one could use "net_aux" while being transparent to
        what exact backbone is used.
        """
        return {"net_aux": "my_custom_arg_name"}

def get_node_na():
    """In your custom python file, this function has to be implemented
    to return the custom algorithm builder"""
    return make_basic_trainer(ModelCustom)
