from libdg.compos.pcr.request import RequestArgs2ExpCmd
from libdg.algos.builder_diva import NodeAlgoBuilderDIVA
from libdg.algos.builder_deepall import NodeAlgoBuilderDeepAll
from libdg.algos.builder_dann import NodeAlgoBuilderDANN
from libdg.algos.builder_hduva import NodeAlgoBuilderHDUVA


class AlgoBuilderChainNodeGetter(object):
    """
    1. Hardcoded chain
    3. Return selected node
    """
    def __init__(self, args):
        self.request = RequestArgs2ExpCmd(args)()

    def __call__(self):
        """
        1. construct the chain, filter out responsible node, create heavy-weight business object
        2. hard code seems to be the best solution
        """
        chain = NodeAlgoBuilderDIVA(None)
        chain = NodeAlgoBuilderDeepAll(chain)
        chain = NodeAlgoBuilderDANN(chain)
        chain = NodeAlgoBuilderHDUVA(chain)
        node = chain.handle(self.request)
        return node
