"""
Base Class for XYD VAE
"""
import torch
import torch.distributions as dist

from domainlab.models.a_model import AModel


class VAEXYD(AModel):
    """
    Base Class for XYD VAE
    """
    def __init__(self, chain_node_builder,
                 zd_dim, zy_dim, zx_dim,
                 list_str_d):
        """
        :param chain_node_builder: constructed object
        """
        super().__init__()
        self.list_str_d = list_str_d
        self.chain_node_builder = chain_node_builder
        self.zd_dim = zd_dim
        self.zy_dim = zy_dim
        self.zx_dim = zx_dim

        self.chain_node_builder.init_business(
            self.zd_dim, self.zx_dim, self.zy_dim)
        self.i_c = self.chain_node_builder.i_c
        self.i_h = self.chain_node_builder.i_h
        self.i_w = self.chain_node_builder.i_w
        self._init_components()

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
