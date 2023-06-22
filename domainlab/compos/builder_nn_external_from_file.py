from domainlab.compos.a_nn_builder import AbstractFeatExtractNNBuilderChainNode
from domainlab.utils.u_import_net_module import \
    build_external_obj_net_module_feat_extract


def mkNodeFeatExtractNNBuilderExternFromFile(arg_name_net_path):
    """
    for each algorithm, there might exist different feature extractors, e.g.
    for diva, there can be class feature extractor and domain feature
    extractor
    """
    class _LNodeFeatExtractNNBuilderExternFromFile(
            AbstractFeatExtractNNBuilderChainNode):
        """LNodeFeatExtractNNBuilderExternFromFile.
        Local class to return
        """
        def init_business(self, dim_out, args, flag_pretrain,
                          remove_last_layer,
                          i_c=None, i_h=None, i_w=None):
            """
            initialize **and** return the heavy weight business object for
            doing the real job
            :param request: subclass can override request object to be
            string or function
            :return: the constructed service object
            """
            pyfile4net = getattr(args, arg_name_net_path)
            net = build_external_obj_net_module_feat_extract(
                   pyfile4net, dim_out, remove_last_layer)
            return net

        def is_myjob(self, args):
            """is_myjob.
            """
            pyfile4net = getattr(args, arg_name_net_path)
            return pyfile4net is not None
    return _LNodeFeatExtractNNBuilderExternFromFile
