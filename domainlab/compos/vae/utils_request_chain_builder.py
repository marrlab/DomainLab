from domainlab.compos.vae.zoo_vae_builders_classif import (
    NodeVAEBuilderArg, NodeVAEBuilderUser, NodeVAEBuilderImgAlex, NodeVAEBuilderImgConvBnPool)
from domainlab.compos.vae.zoo_vae_builders_classif_topic import NodeVAEBuilderImgTopic


class VAEChainNodeGetter(object):
    """
    1. Hardcoded chain, each node use Scenario as request class
    2. Constructor takes parameters for VABuilder Subclasses
    3. heavy weight business objective is returned by selected node
    4. convert Scenario object to request object, so that class can be reused
    """
    def __init__(self, request, topic_dim=None):
        """
        """
        self.request = request
        self.topic_dim = topic_dim

    def __call__(self):
        """
        1. construct the chain, filter out responsible node,
        create heavy-weight business object
        2. hard code seems to be the best solution
        """
        if self.topic_dim is not None:
            chain = NodeVAEBuilderImgTopic(None)
        else:
            chain = NodeVAEBuilderImgConvBnPool(None)
            chain = NodeVAEBuilderImgAlex(chain)
            chain = NodeVAEBuilderArg(chain)
            chain = NodeVAEBuilderUser(chain)
        node = chain.handle(self.request)
        return node
