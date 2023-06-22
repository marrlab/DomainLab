"""
Integrate Chain-of-Responsibility and Builder Patter to construct VAE
encoder and decoder
"""

import abc

from domainlab.compos.pcr.p_chain_handler import AbstractChainNodeHandler
from domainlab.utils.utils_class import store_args


class AbstractVAEBuilderChainNode(AbstractChainNodeHandler):
    """
    to ensure chain of responsibility node AbstractChainNodeHandler always
    work even some node can not start their heavy weight business object,
    avoid override the
    initializer so that node construction is always light weight.
    """
    def __init__(self, successor_node):
        self.args = None
        self.zd_dim = None
        self.zx_dim = None
        self.zy_dim = None
        self.i_c = None
        self.i_h = None
        self.i_w = None
        super().__init__(successor_node)

    @store_args
    def init_business(self, zd_dim, zx_dim, zy_dim):
        """
        initialize **and** return the heavy weight business object for doing the real job
        :param request: subclass can override request object to be string or function
        :return: the constructed service object
        """
        return self

    @abc.abstractmethod
    def build_encoder(self):
        raise NotImplementedError

    @abc.abstractmethod
    def build_decoder(self):
        raise NotImplementedError
