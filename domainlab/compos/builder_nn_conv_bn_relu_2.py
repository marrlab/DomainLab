from domainlab.compos.a_nn_builder import AbstractFeatExtractNNBuilderChainNode
from domainlab.compos.nn_zoo.net_conv_conv_bn_pool_2 import NetConvBnReluPool2L


def mkNodeFeatExtractNNBuilderNameConvBnRelu2(arg_name4net, arg_val,
                                              conv_stride):
    """mkNodeFeatExtractNNBuilderNameConvBnRelu2.
    In chain of responsibility selection of neural network, reuse code to add
    more possibilities of neural network of the same family.
    :param arg_name4net: name of nn in args
    :param arg_val: the registered name of the neural network to be added
    :param conv_stride: should be 1 for 28*28 images
    :param i_c:
    :param i_h:
    :param i_w:
    """
    class _NodeFeatExtractNNBuilderConvBnRelu2L(
            AbstractFeatExtractNNBuilderChainNode):
        """NodeFeatExtractNNBuilderConvBnRelu2L."""

        def init_business(self, dim_out, args, i_c, i_h, i_w,
                          flag_pretrain=None, remove_last_layer=False):
            """
            :param flag_pretrain
            """
            self.net_feat_extract = NetConvBnReluPool2L(
                i_c=i_c, i_h=i_h, i_w=i_w,
                conv_stride=conv_stride, dim_out_h=dim_out)
            return self.net_feat_extract

        def is_myjob(self, args):
            """is_myjob.
            :param args: command line arguments:
                "--nname": name of the torchvision model
            """
            arg_name = getattr(args, arg_name4net)
            return arg_name == arg_val
    return _NodeFeatExtractNNBuilderConvBnRelu2L
