"""
Base Class for XYD VAE Classify
"""
from domainlab.models.a_model_classif import AModelClassif
from domainlab.models.model_vae_xyd import VAEXYD
from domainlab.utils.utils_class import store_args


class VAEXYDClassif(AModelClassif, VAEXYD):
    """
    Base Class for DIVA and HDUVA
    """
    @store_args
    def __init__(self, chain_node_builder,
                 zd_dim, zy_dim, zx_dim,
                 list_str_y, list_str_d):
        """
        :param chain_node_builder: constructed object
        """
        VAEXYD.__init__(self, chain_node_builder, zd_dim, zy_dim, zx_dim, list_str_y, list_str_d)
        AModelClassif.__init__(self, list_str_y, list_str_d)

    def cal_logit_y(self, tensor_x):
        """
        calculate the logit for softmax classification
        """
        zy_q_loc = self.encoder.infer_zy_loc(tensor_x)
        logit_y = self.net_classif_y(zy_q_loc)
        return logit_y

    def _init_components(self):
        super()._init_components()
        self.add_module("net_classif_y",
                        self.chain_node_builder.construct_classifier(
                            self.zy_dim, self.dim_y))
