"""
interface for observer + visitor
"""
import abc


class AObVisitor(metaclass=abc.ABCMeta):
    """
    Observer + Visitor pattern for model selection
    """
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
