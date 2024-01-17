"""
Multiobjective Model Selection
"""
import copy

from domainlab.algos.msels.a_model_sel import AMSel
from domainlab.utils.logger import Logger


class MSelSetpointDelay(AMSel):
    """
    1. Model selection using validation performance
    2. Only update if setpoint has been decreased
    """

    def __init__(self, msel):
        super().__init__()
        # NOTE: super() has to come first always otherwise self.msel will be overwritten to be None
        self.msel = msel
        self._oracle_last_setpoint_sel_te_acc = 0.0

    @property
    def oracle_last_setpoint_sel_te_acc(self):
        """
        return the last setpoint best acc
        """
        return self._oracle_last_setpoint_sel_te_acc

    def update(self, clear_counter=False):
        """
        if the best model should be updated
        """
        logger = Logger.get_logger()
        logger.info(
            f"setpoint selected current acc {self._oracle_last_setpoint_sel_te_acc}"
        )
        if clear_counter:
            logger.info(
                "setpoint msel te acc updated from {self._oracle_last_setpoint_sel_te_acc} to {self.sel_model_te_acc}"
            )
            self._oracle_last_setpoint_sel_te_acc = self.sel_model_te_acc
        flag = self.msel.update(clear_counter)
        return flag
