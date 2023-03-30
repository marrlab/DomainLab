from domainlab.algos.builder_dann import NodeAlgoBuilderDANN
from domainlab.algos.builder_jigen1 import NodeAlgoBuilderJiGen
from domainlab.algos.builder_deepall import NodeAlgoBuilderDeepAll
from domainlab.algos.builder_dial import NodeAlgoBuilderDeepAll_DIAL
from domainlab.algos.builder_deepall_mldg import NodeAlgoBuilderDeepAllMLDG
from domainlab.algos.builder_diva import NodeAlgoBuilderDIVA
from domainlab.algos.builder_hduva import NodeAlgoBuilderHDUVA
from domainlab.algos.builder_matchdg import NodeAlgoBuilderMatchDG
from domainlab.algos.builder_match_hduva import NodeAlgoBuilderMatchHDUVA
from domainlab.compos.pcr.request import RequestArgs2ExpCmd
from domainlab.utils.u_import import import_path


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
        node_fun = node_module.get_node_na()  # @FIXME: build_node API need
        newchain = node_fun(chain)
        return newchain

    def __call__(self):
        """
        1. construct the chain, filter out responsible node, create heavy-weight business object
        2. hard code seems to be the best solution
        """
        chain = NodeAlgoBuilderDIVA(None)
        chain = NodeAlgoBuilderDeepAll(chain)
        chain = NodeAlgoBuilderDeepAll_DIAL(chain)
        chain = NodeAlgoBuilderDeepAllMLDG(chain)
        chain = NodeAlgoBuilderDANN(chain)
        chain = NodeAlgoBuilderJiGen(chain)
        chain = NodeAlgoBuilderHDUVA(chain)
        chain = NodeAlgoBuilderMatchDG(chain)
        chain = NodeAlgoBuilderMatchHDUVA(chain)
        chain = self.register_external_node(chain)
        node = chain.handle(self.request)
        return node
