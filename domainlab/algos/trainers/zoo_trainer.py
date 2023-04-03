"""
select trainer
"""
from domainlab.algos.trainers.train_basic import TrainerBasic
from domainlab.algos.trainers.train_dial import TrainerDIAL
from domainlab.algos.trainers.train_matchdg import TrainerMatchDG
from domainlab.algos.trainers.train_mldg import TrainerMLDG
from domainlab.algos.trainers.train_visitor import TrainerVisitor


class TrainerChainNodeGetter(object):
    """
    1. Hardcoded chain
    3. Return selected node
    """
    def __init__(self, args):
        """__init__.
        :param args: command line arguments
        """
        self.request = args

    def __call__(self, lst_candidates):
        """
        1. construct the chain, filter out responsible node,
        create heavy-weight business object
        2. hard code seems to be the best solution
        """
        chain = TrainerBasic(None)
        chain = TrainerDIAL(chain)
        chain = TrainerMatchDG(chain)
        chain = TrainerMLDG(chain)
        chain = TrainerVisitor(chain)
        node = chain.handle(self.request)
        return node
