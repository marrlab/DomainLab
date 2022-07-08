from libdg.compos.a_nn_builder import AbstractFeatExtractNNBuilderChainNode
from libdg.compos.nn_alex import Alex4DeepAll, AlexNetNoLastLayer


class NodeFeatExtractNNBuilderAlex(AbstractFeatExtractNNBuilderChainNode):
    def init_business(self, flag_pretrain, dim_feat, remove_last_layer=False):
        """
        initialize **and** return the heavy weight business object for doing
        the real job
        :param request: subclass can override request object to be string or
        function
        :return: the constructed service object
        """
        self.net_feat_extract = Alex4DeepAll(flag_pretrain, dim_feat)
        if remove_last_layer:
            self.net_feat_extract = Alex4DeepAll(flag_pretrain, dim_feat)
        return self.net_feat_extract = AlexNetNoLastLayer(flag_pretrain)

    def is_myjob(self, args):
        """is_myjob.
        :param args_nname: command line arguments: "--nname": \
            name of the torchvision model
        """
        if args.npath is not None:
            return False
        return args.nname == "alexnet"
