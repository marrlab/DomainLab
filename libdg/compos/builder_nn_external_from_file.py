from libdg.compos.a_nn_builder import AbstractFeatExtractNNBuilderChainNode
from libdg.utils.u_import_net_module import \
    build_external_obj_net_module_feat_extract


class NodeFeatExtractNNBuilderExternFromFile(AbstractFeatExtractNNBuilderChainNode):
    def init_business(self, flag_pretrain, dim_y, remove_last_layer, args):
        """
        initialize **and** return the heavy weight business object for doing
        the real job
        :param request: subclass can override request object to be string or
        function
        :return: the constructed service object
        """
        net = build_external_obj_net_module_feat_extract(
                args.npath, dim_y, remove_last_layer)
        return net
