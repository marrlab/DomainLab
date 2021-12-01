from libdg.compos.vae.zoo_vae_builders_classif import NodeVAEBuilderImg28, NodeVAEBuilderImg64, NodeVAEBuilderImg224
from libdg.compos.vae.zoo_vae_builders_classif import NodeVAEBuilderImg224Topic
from libdg.utils.utils_class import store_args
from libdg.compos.pcr.request import RequestVAEBuilderCHW


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
        1. construct the chain, filter out responsible node, create heavy-weight business object
        2. hard code seems to be the best solution
        """
        if self.topic_dim is not None:
            chain = NodeVAEBuilderImg224Topic(None)
        else:
            chain = NodeVAEBuilderImg28(None)
            chain = NodeVAEBuilderImg64(chain)
            chain = NodeVAEBuilderImg224(chain)
        node = chain.handle(self.request)
        return node


def test_VAEChainBuilderScenarioAsRequest():
    request = RequestVAEBuilderCHW(3, 64, 64)
    node = VAEChainNodeGetter(request)()
    builder = node.init_business(8, 8, 8)
    encoder = builder.build_encoder()
    decoder = builder.build_decoder()
