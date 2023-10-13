"""
Multiobjective Model Selection
"""
import copy
from domainlab.algos.msels.a_model_sel import AMSel


class MSelSetpointDelay(AMSel):
    """
    1. Model selection using validation performance
    2. Only update if setpoint has been decreased
    """
    def __init__(self, msel):
        super().__init__()
        self.msel = msel
        self.tr_obs = msel.tr_obs
        self.trainer = msel.trainer
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
        if clear_counter:
            self._oracle_last_setpoint_sel_te_acc = self.sel_model_te_acc
        flag = self.msel.update(clear_counter)
        return flag
