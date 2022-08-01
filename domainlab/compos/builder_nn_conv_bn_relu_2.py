from domainlab.compos.a_nn_builder import AbstractFeatExtractNNBuilderChainNode
from domainlab.compos.net_conv import NetConvBnReluPool2


def mkNodeFeatExtractNNBuilderNameConvBnRelu2(arg_name4net):
    class NodeFeatExtractNNBuilderConvBnRelu2(
            AbstractFeatExtractNNBuilderChainNode):
        """NodeFeatExtractNNBuilderAlex.
        Uniform interface to return AlexNet and other neural network as feature
        extractor from torchvision or external python file"""
        # FIXME: the signature of init_business is different from others
        def init_business(self, dim_out_h, i_c, i_h, i_w, conv_stride):
            """
            initialize **and** return the heavy weight business
            object for doing the real job
            :param request: subclass can override request object
            to be string or function
            :return: the constructed service object
            """
            self.net_feat_extract = NetConvBnReluPool2(
                i_c, i_h, i_w, conv_stride, dim_out_h)
            return self.net_feat_extract

        def is_myjob(self, args):
            """is_myjob.
            :param args: command line arguments:
                "--nname": name of the torchvision model
            """
            arg_name = getattr(args, arg_name4net)
            return arg_name == "conv_bn_pool_2"   # FIXME
    return NodeFeatExtractNNBuilderConvBnRelu2
