import copy
from libdg.compos.builder_nn_alex import NodeFeatExtractNNBuilderAlex
from libdg.compos.builder_nn_external_from_file import \
    NodeFeatExtractNNBuilderExternFromFile


class FeatExtractNNBuilderChainNodeGetter(object):
    """
    1. Hardcoded chain
    3. Return selected node
    """
    def __init__(self, args, task):
        self.request = args
        self.args = args
        self.task = task

    def __call__(self):
        """
        1. construct the chain, filter out responsible node,
        create heavy-weight business object
        2. hard code seems to be the best solution
        """
        chain = NodeFeatExtractNNBuilderAlex(None)
        if self.args.npath is None:
            node = chain.handle(self.request)
        else:
            node = NodeFeatExtractNNBuilderExternFromFile(None)
        return node
