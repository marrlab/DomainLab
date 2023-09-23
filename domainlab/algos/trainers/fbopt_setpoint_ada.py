"""
update hyper-parameters during training
"""
import copy
import numpy as np
import torch

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
    def __init__(self):
        """
        kwargs is a dictionary with key the hyper-parameter name and its value
        """
        self.ma_epo_reg_loss = None
        self.state_epo_reg_loss = None
        self.coeff_ma = None
        self.state_updater = None
        self.setpoint4R = None
        self.setpoint4ell = None

    def update_setpoint_ma(self, target):
        temp_ma = self.coeff_ma * torch.tensor(target)
        temp_ma += (1 - self.coeff_ma) * torch.tensor(self.setpoint4R)
        temp_ma = temp_ma.tolist()
        self.setpoint4R = temp_ma

    def observe(self, epo_reg_loss):
        """
        read current epo_reg_loss continuously
        FIXME: setpoint should also be able to be eliviated
        """
        self.state_epo_reg_loss = epo_reg_loss
        self.state_updater.update_setpoint()
        if self.state_updater.update_setpoint():
            logger = Logger.get_logger(logger_name='main_out_logger', loglevel="INFO")
            logger.info("!!!!!set point updated to {self.setpoint4R}!")


class SliderAll(FbOptSetpointController):
    def update_setpoint(self):
        if is_less_list_all(self.state_epo_reg_loss, self.setpoint4R):
            self.setpoint4R = self.state_epo_reg_loss
            return True
        return False


class SliderAny(FbOptSetpointController):
    def update_setpoint(self):
        if is_less_list_all(self.state_epo_reg_loss, self.setpoint4R):
            self.setpoint4R = self.state_epo_reg_loss
            return True
        return False

class DominateAny(FbOptSetpointController):
    def update_setpoint(self):
        flag1 = is_less_list_any(self.state_epo_reg_loss, self.setpoint4R)
        flag2 = self.state_task_loss < self.setpoint4ell
        return flag1 & flag2

class DominateAll(FbOptSetpointController):
    def update_setpoint(self):
        flag1 = is_less_list_all(self.state_epo_reg_loss, self.setpoint4R)
        flag2 = self.state_task_loss < self.setpoint4ell
        return flag1 & flag2
