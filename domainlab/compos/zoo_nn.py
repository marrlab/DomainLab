import copy
from domainlab.compos.builder_nn_alex import mkNodeFeatExtractNNBuilderNameAlex
from domainlab.compos.builder_nn_external_from_file import \
    mkNodeFeatExtractNNBuilderExternFromFile


class FeatExtractNNBuilderChainNodeGetter(object):
    """
    1. Hardcoded chain
    3. Return selected node
    """
    def __init__(self, args, arg_name_of_net,
                 arg_path_of_net):
        """__init__.
        :param args: command line arguments
        :param arg_name_of_net: args.npath to specify
        where to get the external architecture for example
        """
        self.request = args
        self.arg_name_of_net = arg_name_of_net
        self.arg_path_of_net = arg_path_of_net

    def __call__(self):
        """
        1. construct the chain, filter out responsible node,
        create heavy-weight business object
        2. hard code seems to be the best solution
        """
        chain = mkNodeFeatExtractNNBuilderNameAlex(
            self.arg_name_of_net)(None)
        chain = mkNodeFeatExtractNNBuilderExternFromFile(
            self.arg_path_of_net)(chain)
        node = chain.handle(self.request)
        return node
