"""
interface for observer + visitor
"""
import abc


class AObVisitor(metaclass=abc.ABCMeta):
    """
    Observer + Visitor pattern for model selection
    """
    def __init__(self):
        self.exp = None
        self.task = None
        self.epo_te = None
        self.str_msel = None
        self.keep_model = None
        self.loader_te = None
        self.loader_tr = None
        self.loader_val = None

    @abc.abstractmethod
    def update(self, epoch) -> bool:
        """
        return True/False whether the trainer should stop training
        """

    @abc.abstractmethod
    def accept(self, trainer):
        """
        accept invitation as a visitor
        """

    @abc.abstractmethod
    def after_all(self):
        """
        After training is done
        """

    @abc.abstractmethod
    def clean_up(self):
        """
        to be called by a decorator
        """
