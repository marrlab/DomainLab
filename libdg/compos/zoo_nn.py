import copy
from libdg.compos.builder_nn_alex import NodeFeatExtractNNBuilderAlex
from libdg.compos.builder_nn_external_from_file import \
    mkNodeFeatExtractNNBuilderExternFromFile


class FeatExtractNNBuilderChainNodeGetter(object):
    """
    1. Hardcoded chain
    3. Return selected node
    """
    def __init__(self, args, arg_name_of_net):
        """__init__.
        :param args: command line arguments
        :param arg_name_of_net: args.npath to specify
        where to get the external architecture for example
        """
        self.request = args
        self.arg_name_of_net = arg_name_of_net

    def __call__(self):
        """
        1. construct the chain, filter out responsible node,
        create heavy-weight business object
        2. hard code seems to be the best solution
        """
        chain = NodeFeatExtractNNBuilderAlex(None)
        chain = mkNodeFeatExtractNNBuilderExternFromFile(
            self.arg_name_of_net)(chain)
        node = chain.handle(self.request)
        return node
