"""
parent class for combing model, trainer, task, observer
"""
import abc
from domainlab.compos.pcr.p_chain_handler import AbstractChainNodeHandler
from domainlab.utils.logger import Logger


class NodeAlgoBuilder(AbstractChainNodeHandler):
    """
    Base class for Algorithm Builder
    """
    na_prefix = "NodeAlgoBuilder"

    @property
    def name(self):
        """
        get the name of the algorithm
        """
        na_prefix = NodeAlgoBuilder.na_prefix
        len_prefix = len(na_prefix)
        na_class = type(self).__name__
        if na_class[:len_prefix] != na_prefix:
            raise RuntimeError(
                "algorithm builder node class must start with ", na_prefix,
                "the current class is named: ", na_class)
        return type(self).__name__[len_prefix:].lower()

    def is_myjob(self, request):
        """
        :param request: string
        """
        return request == self.name

    @abc.abstractmethod
    def init_business(self, exp):
        """
        combine model, trainer, observer, task
        """
