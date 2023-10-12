"""
Abstract Model Selection
"""

import abc


class AMSel(metaclass=abc.ABCMeta):
    """
    Abstract Model Selection
    """
    def __init__(self):
        """
        trainer and tr_observer
        """
        self.trainer = None
        self.tr_obs = None

    def accept(self, trainer, tr_obs):
        """
        Visitor pattern to trainer
        accept trainer and tr_observer
        """
        self.trainer = trainer
        self.tr_obs = tr_obs

    @abc.abstractmethod
    def update(self, clear_counter=False):
        """
        observer + visitor pattern to trainer
        if the best model should be updated
        return boolean
        """

    def if_stop(self):
        """
        check if trainer should stop
        return boolean
        """
        raise NotImplementedError
