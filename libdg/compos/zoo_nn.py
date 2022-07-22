import copy
from libdg.compos.builder_nn_alex import NodeFeatExtractNNBuilderAlex
from libdg.compos.builder_nn_external_from_file import \
    NodeFeatExtractNNBuilderExternFromFile


class FeatExtractNNBuilderChainNodeGetter(object):
    """
    1. Hardcoded chain
    3. Return selected node
    """
    def __init__(self, args):
        self.request = args

    def __call__(self):
        """
        1. construct the chain, filter out responsible node,
        create heavy-weight business object
        2. hard code seems to be the best solution
        """
        chain = NodeFeatExtractNNBuilderAlex(None)
        chain = NodeFeatExtractNNBuilderExternFromFile(chain)
        node = chain.handle(self.request)
        return node
