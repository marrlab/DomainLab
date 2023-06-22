"""
Integrate Chain-of-Responsibility and Builder Patter to construct VAE encoder and decoder
"""

import abc

from domainlab.compos.pcr.p_chain_handler import AbstractChainNodeHandler


class AbstractModelBuilderChainNode(AbstractChainNodeHandler):
    """
    to ensure chain of responsibility node AbstractChainNodeHandler always
    work even some node can not start their heavy weight business object, avoid override the
    initializer so that node construction is always light weight.

    The config() method here is abstract, while child class has a concrete config method
    """

    @abc.abstractmethod
    def config(self, *kargs, **kwargs):
        """
        use either list or dictionary input arguments to configure the model builder.
        """
        raise NotImplementedError
