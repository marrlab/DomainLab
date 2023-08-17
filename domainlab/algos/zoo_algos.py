"""
chain of responsibility pattern for algorithm selection
"""
from domainlab.algos.builder_dann import NodeAlgoBuilderDANN
from domainlab.algos.builder_jigen1 import NodeAlgoBuilderJiGen
from domainlab.algos.builder_deepall import NodeAlgoBuilderDeepAll
from domainlab.algos.builder_diva import NodeAlgoBuilderDIVA
from domainlab.algos.builder_hduva import NodeAlgoBuilderHDUVA
from domainlab.algos.builder_matchdg import NodeAlgoBuilderMatchDG
from domainlab.algos.builder_match_hduva import NodeAlgoBuilderMatchHDUVA
from domainlab.algos.builder_api_model import NodeAlgoBuilderAPIModel

from domainlab.utils.u_import import import_path


class AlgoBuilderChainNodeGetter():
    """
    1. Hardcoded chain
    3. Return selected node
    """
    def __init__(self, aname, apath):
        self.aname = aname
        self.apath = apath

    def register_external_node(self, chain):
        """
        if the user specify an external python file to implement the algorithm
        """
        if self.apath is None:
            return chain
        node_module = import_path(self.apath)
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
        chain = NodeAlgoBuilderDANN(chain)
        chain = NodeAlgoBuilderJiGen(chain)
        chain = NodeAlgoBuilderHDUVA(chain)
        chain = NodeAlgoBuilderMatchDG(chain)
        chain = NodeAlgoBuilderMatchHDUVA(chain)
        chain = NodeAlgoBuilderAPIModel(chain)
        chain = self.register_external_node(chain)
        node = chain.handle(self.aname)
        return node
