"""
Abstract Model Selection
"""

import abc


class AMSel(metaclass=abc.ABCMeta):
    """
    Abstract Model Selection
    """

    def __init__(self, val_threshold = None):
        """
        trainer and tr_observer
        """
        self.trainer = None
        self._observer = None
        self.msel = None
        self._max_es = None
        self._model_selection_epoch = None
        self._val_threshold = val_threshold

    def reset(self):
        """
        reset observer via reset model selector
        """
        if self.msel is not None:
            self.msel.reset()

    @property
    def observer4msel(self):
        """
        the observer from trainer
        """
        return self._observer

    @property
    def max_es(self):
        """
        maximum early stop
        """
        if self._max_es is not None:
            return self._max_es
        if self.msel is not None:
            return self.msel.max_es
        return self._max_es

    def accept(self, trainer, observer4msel):
        """
        Visitor pattern to trainer
        accept trainer and tr_observer
        """
        self.trainer = trainer
        self._observer = observer4msel
        if self.msel is not None:
            self.msel.accept(trainer, observer4msel)

    def update(self, epoch, clear_counter=False):
        """
        level above the observer + visitor pattern to get information about the epoch
        """
        update = self.base_update(clear_counter)
        if update:
            self._model_selection_epoch = epoch

        return update

    @abc.abstractmethod
    def base_update(self, clear_counter=False):
        """
        observer + visitor pattern to trainer
        if the best model should be updated
        return boolean
        """

    def if_stop(self, acc_val = None):
        """
        check if trainer should stop and additionally tests for validation threshold
        return boolean
        """
        # NOTE: since if_stop is not abstract, one has to
        # be careful to always override it in child class
        # only if the child class has a decorator which will
        # dispatched.
        if self.msel is not None and acc_val is not None:
            if self._val_threshold is not None and acc_val < self._val_threshold:
                return False
        return self.early_stop()

    def early_stop(self):
        """
        check if trainer should stop
        return boolean
        """
        if self.msel is not None:
            return self.msel.early_stop()
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
        """
        the selected model test accuaracy
        """
        if self.msel is not None:
            return self.msel.sel_model_te_acc
        return -1

    @property
    def oracle_last_setpoint_sel_te_acc(self):
        if self.msel is not None:
            return self.msel.oracle_last_setpoint_sel_te_acc
        return -1

    @property
    def model_selection_epoch(self):
        """
        the epoch when the model was selected
        """
        if self._model_selection_epoch is not None:
            return self._model_selection_epoch
        return -1

    @property
    def val_threshold(self):
        """
        the treshold below which we don't stop early
        """
        return self._val_threshold
