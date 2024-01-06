"""
select trainer
"""
from domainlab.algos.trainers.train_basic import TrainerBasic
from domainlab.algos.trainers.train_dial import TrainerDIAL
from domainlab.algos.trainers.train_matchdg import TrainerMatchDG
from domainlab.algos.trainers.train_mldg import TrainerMLDG
from domainlab.algos.trainers.train_hyper_scheduler import TrainerHyperScheduler


class TrainerChainNodeGetter(object):
    """
    Chain of Responsibility: node is named in pattern Trainer[XXX] where the string
    after 'Trainer' is the name to be passed to args.trainer.
    """
    def __init__(self, str_trainer):
        """__init__.
        :param args: command line arguments
        """
        self._list_str_trainer = None
        if str_trainer is not None:
            self._list_str_trainer = str_trainer.split(',')
            self.request = self._list_str_trainer.pop(0)
        else:
            self.request = str_trainer

    def __call__(self, lst_candidates=None, default=None, lst_excludes=None):
        """
        1. construct the chain, filter out responsible node,
        create heavy-weight business object
        2. hard code seems to be the best solution
        """
        if lst_candidates is not None and self.request not in lst_candidates:
            raise RuntimeError(f"desired {self.request} is not supported \
                               among {lst_candidates}")
        if default is not None and self.request is None:
            self.request = default
        if lst_excludes is not None and self.request in lst_excludes:
            raise RuntimeError(f"desired {self.request} is not supported among {lst_excludes}")

        chain = TrainerBasic(None)
        chain = TrainerDIAL(chain)
        chain = TrainerMatchDG(chain)
        chain = TrainerMLDG(chain)
        chain = TrainerHyperScheduler(chain)
        node = chain.handle(self.request)
        head = node
        while self._list_str_trainer:
            self.request = self._list_str_trainer.pop(0)
            node2decorate = self.__call__(lst_candidates, default, lst_excludes)
            head.extend(node2decorate)
            head = node2decorate
        return node
