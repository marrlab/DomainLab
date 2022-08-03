from domainlab.compos.a_nn_builder import AbstractFeatExtractNNBuilderChainNode
from domainlab.compos.nn_zoo.nn_alex import Alex4DeepAll, AlexNetNoLastLayer


def mkNodeFeatExtractNNBuilderNameAlex(arg_name4net, arg_val):
    class NodeFeatExtractNNBuilderAlex(AbstractFeatExtractNNBuilderChainNode):
        """NodeFeatExtractNNBuilderAlex.
        Uniform interface to return AlexNet and other neural network as feature
        extractor from torchvision or external python file"""
        def init_business(self, dim_out, args, i_c=None, i_h=None, i_w=None,
                          remove_last_layer=False, flag_pretrain=True):
            """
            initialize **and** return the heavy weight business
            object for doing the real job
            :param request: subclass can override request object
            to be string or function
            :return: the constructed service object
            """
            if not remove_last_layer:
                self.net_feat_extract = Alex4DeepAll(flag_pretrain, dim_out)
            else:
                self.net_feat_extract = AlexNetNoLastLayer(flag_pretrain)
            return self.net_feat_extract

        def is_myjob(self, args):
            """is_myjob.
            :param args: command line arguments:
                "--nname": name of the torchvision model
            """
            arg_name = getattr(args, arg_name4net)
            return arg_name == arg_val
    return NodeFeatExtractNNBuilderAlex
