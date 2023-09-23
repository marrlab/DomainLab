"""
update hyper-parameters during training
"""
import copy
import numpy as np
from domainlab.utils.logger import Logger


def is_less_list_any(list1, list2):
    """
    judge if one list is less than the other
    """
    list_comparison = [a < b for a, b in zip(list1, list2)]
    return any(list_comparison)


def is_less_list_all(list1, list2):
    """
    judge if one list is less than the other
    """
    list_comparison = [a < b for a, b in zip(list1, list2)]
    return all(list_comparison)




class FbOptSetpointController():
    """
    design $\\mu$$ sequence based on state of penalized loss
    """
    def __init__(self, trainer, **kwargs):
        """
        kwargs is a dictionary with key the hyper-parameter name and its value
        """
        self.ma_epo_reg_loss = None
        self.state_epo_reg_loss = None


    def observe(self, epo_reg_loss):
        """
        FIXME: setpoint should also be able to be eliviated
        """
        # FIXME: what does smaller than mean for a list?
        # FIXME: use pareto-reg-descent operator to decide if set point should be adjusted
        if epo_reg_loss < self.setpoint4R:
            logger = Logger.get_logger(logger_name='main_out_logger', loglevel="INFO")
            lower_bound = self.coeff_ma * torch.tensor(epo_reg_loss)
            lower_bound += (1-self.coeff_ma) * torch.tensor(self.setpoint4R)
            lower_bound = lower_bound.tolist()
            self.setpoint4R = lower_bound
            logger.info("!!!!!set point updated to {lower_bound}!")


class FbOptSetpointAdaRAllComponent(FbOptSetpointController):
    def update_setpoint(self):
        if is_less_list_all(self.state_epo_reg_loss, self.setpoint4R):
            self.setpoint4R = self.state_epo_reg_loss
