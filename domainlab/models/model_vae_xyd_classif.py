"""
Base Class for XYD VAE
"""
import torch
import torch.distributions as dist

from domainlab.models.a_model_classif import AModelClassif
from domainlab.utils.utils_class import store_args


class VAEXYDClassif(AModelClassif):
    """
    Base Class for XYD VAE
    """
    @store_args
    def __init__(self, chain_node_builder,
                 zd_dim, zy_dim, zx_dim,
                 list_str_y, list_str_d):
        """
        :param chain_node_builder: constructed object
        """
        super().__init__(list_str_y, list_str_d)
        self.chain_node_builder.init_business(
            self.zd_dim, self.zx_dim, self.zy_dim)
        self.i_c = self.chain_node_builder.i_c
        self.i_h = self.chain_node_builder.i_h
        self.i_w = self.chain_node_builder.i_w
        self._init_components()

    def cal_logit_y(self, tensor_x):
        """
        calculate the logit for softmax classification
        """
        zy_q_loc = self.encoder.infer_zy_loc(tensor_x)
        logit_y = self.net_classif_y(zy_q_loc)
        return logit_y

    def _init_components(self):
        """
        q(z|x)
        p(zy)
        q_{classif}(zy)
        """
        self.add_module("encoder", self.chain_node_builder.build_encoder())
        self.add_module("decoder", self.chain_node_builder.build_decoder())
        self.add_module("net_p_zy",
                        self.chain_node_builder.construct_cond_prior(
                            self.dim_y, self.zy_dim))
        self.add_module("net_classif_y",
                        self.chain_node_builder.construct_classifier(
                            self.zy_dim, self.dim_y))

    def init_p_zx4batch(self, batch_size, device):
        """
        1. Generate pytorch distribution object.
        2. To be called by trainer

        :param batch_size:
        :param device:
        """
        # p(zx): isotropic gaussian
        zx_p_loc = torch.zeros(batch_size, self.zx_dim).to(device)
        zx_p_scale = torch.ones(batch_size, self.zx_dim).to(device)
        p_zx = dist.Normal(zx_p_loc, zx_p_scale)
        return p_zx
