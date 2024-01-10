"""
Base Class for XYD VAE Classify
"""
from domainlab.models.a_model_classif import AModelClassif
from domainlab.models.interface_vae_xyd import InterfaceVAEXYD
from domainlab.utils.utils_class import store_args


class VAEXYDClassif(AModelClassif, InterfaceVAEXYD):
    """
    Base Class for DIVA and HDUVA
    """
    @store_args
    def __init__(self, chain_node_builder,
                 zd_dim, zy_dim, zx_dim,
                 list_str_y):
        """
        :param chain_node_builder: constructed object
        """
        super().__init__(list_str_y)
        self.init()

    def extract_semantic_feat(self, tensor_x):
        zy_q_loc = self.encoder.infer_zy_loc(tensor_x)
        return zy_q_loc

    @property
    def multiplier4task_loss(self):
        """
        the multiplier for task loss is default to 1.0 except for vae family models
        """
        return self.gamma_y

    def _init_components(self):
        super()._init_components()
        self.add_module("net_classif_y",
                        self.chain_node_builder.construct_classifier(
                            self.zy_dim, self.dim_y))
        # property setter only for other object, internally, one shoud use _net_classifier
        self._net_classifier = self.net_classif_y
