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
        self.msel = None

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

    @property
    def best_val_acc(self):
        """
        decoratee best val acc
        """
        if self.msel is not None:
            return self.msel.best_val_acc
        return -1

    @property
    def best_te_metric(self):
        """
        decoratee best test metric
        """
        if self.msel is not None:
            return self.msel.best_te_metric
        return -1
        
    @property
    def sel_model_te_acc(self):
        if self.msel is not None:
            return self.msel.sel_model_te_acc
        return -1
        
    @property
    def oracle_last_setpoint_sel_te_acc(self):
        if self.msel is not None:
            return self.msel.oracle_last_setpoint_sel_te_acc
        return -1