"""
Multiobjective Model Selection
"""
import copy
from domainlab.algos.msels.c_msel_val import MSelValPerf


class MSelSetpointDelay(MSelValPerf):
    """
    1. Model selection using validation performance
    2. Only update if setpoint has been decreased
    """
    def __init__(self, msel):
        self.msel = copy.deepcopy(msel)
        self._oracle_last_setpoint_sel_te_acc = 0.0
        super().__init__(msel.max_es)
        
    @property
    def self.oracle_last_setpoing_sel_te_acc(self):
        return self._oracle_last_setpoint_sel_te_acc
    
    def update(self, clear_counter=False):
        """
        if the best model should be updated
        """
        if clear_counter:
            self._oracle_last_setpoint_sel_te_acc = self.sel_model_te_acc
        flag = self.msel.update(clear_counter)
        # FIXME: flag is to persist model, which is not possible anymore
        return flag
