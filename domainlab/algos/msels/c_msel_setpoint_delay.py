"""
logs the best up-to-event selected model at each event when setpoint shrinks
"""
from domainlab.algos.msels.a_model_sel import AMSel
from domainlab.utils.logger import Logger


class MSelSetpointDelay(AMSel):
    """
    This class decorate another model selection object, it logs the current
    selected performance from the decoratee each time the setpoint shrinks
    """

    def __init__(self, msel, val_threshold = None):
        super().__init__(val_threshold)
        # NOTE: super() has to come first always otherwise self.msel will be overwritten to be None
        self.msel = msel
        self._oracle_last_setpoint_sel_te_acc = 0.0

    @property
    def oracle_last_setpoint_sel_te_acc(self):
        """
        return the last setpoint best acc
        """
        return self._oracle_last_setpoint_sel_te_acc

    def base_update(self, clear_counter=False):
        """
        if the best model should be updated
        currently, clear_counter is set via
        flag = super().tr_epoch(epoch, self.flag_setpoint_updated)
        """
        logger = Logger.get_logger()
        logger.info(
            f"setpoint selected current acc {self._oracle_last_setpoint_sel_te_acc}"
        )
        if clear_counter:
            # for the current version of code, clear_counter = flag_setpoint_updated
            log_message = (
                f"setpoint msel te acc updated from "
                # self._oracle_last_setpoint_sel_te_acc start from 0.0, and always saves
                # the test acc when last setpoint decrease occurs
                f"{self._oracle_last_setpoint_sel_te_acc} to "
                # self.sel_model_te_acc defined as a property
                # in a_msel, which returns self.msel.sel_model_te_acc
                # is the validation acc based model selection, which
                # does not take setpoint into account
                f"{self.sel_model_te_acc}"
            )
            logger.info(log_message)
            self._oracle_last_setpoint_sel_te_acc = self.sel_model_te_acc
        # let decoratee decide if model should be selected or not
        flag = self.msel.update(clear_counter)
        return flag
