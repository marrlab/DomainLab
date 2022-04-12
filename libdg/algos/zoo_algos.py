from libdg.compos.pcr.request import RequestArgs2ExpCmd
from libdg.algos.builder_diva import NodeAlgoBuilderDIVA
from libdg.algos.builder_deepall import NodeAlgoBuilderDeepAll
from libdg.algos.builder_dann import NodeAlgoBuilderDANN
from libdg.algos.builder_hduva import NodeAlgoBuilderHDUVA
from libdg.algos.builder_matchdg import NodeAlgoBuilderMatchDG
from libdg.utils.u_import import import_path


class AlgoBuilderChainNodeGetter(object):
    """
    1. Hardcoded chain
    3. Return selected node
    """
    def __init__(self, args):
        self.request = RequestArgs2ExpCmd(args)()
        self.args = args

    def register_external_node(self, chain):
        if self.args.apath is None:
            return chain
        node_module = import_path(self.args.apath)
        node_fun = node_module.build_node()  # FIXME: build_node API need
        newchain = node_fun()(chain)
        return newchain

    def __call__(self):
        """
        1. construct the chain, filter out responsible node, create heavy-weight business object
        2. hard code seems to be the best solution
        """
        chain = NodeAlgoBuilderDIVA(None)
        chain = NodeAlgoBuilderDeepAll(chain)
        chain = NodeAlgoBuilderDANN(chain)
        chain = NodeAlgoBuilderHDUVA(chain)
        chain = NodeAlgoBuilderMatchDG(chain)
        chain = self.register_external_node(chain)
        node = chain.handle(self.request)
        return node
