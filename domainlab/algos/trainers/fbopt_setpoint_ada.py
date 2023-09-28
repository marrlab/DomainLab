"""
update hyper-parameters during training
"""
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
    update setpoint for mu
    """
    def __init__(self, state=None):
        """
        kwargs is a dictionary with key the hyper-parameter name and its value
        """
        if state is None:
            state = SliderAnyComponent()
        self.transition_to(state)
        self.ma_epo_reg_loss = None
        self.state_epo_reg_loss = None
        self.coeff_ma = 0.5  # FIXME
        self.state_task_loss = None
        # initial value will be set via trainer
        self.setpoint4R = None
        self.setpoint4ell = None
        self.host = None

    def transition_to(self, state):
        """
        change internal state
        """
        self.state_updater = state
        self.state_updater.accept(self)

    def update_setpoint_ma(self, target):
        """
        using moving average
        """
        temp_ma = self.coeff_ma * torch.tensor(target)
        temp_ma += (1 - self.coeff_ma) * torch.tensor(self.setpoint4R)
        temp_ma = temp_ma.tolist()
        self.setpoint4R = temp_ma

    def observe(self, epo_reg_loss, epo_task_loss):
        """
        read current epo_reg_loss continuously
        FIXME: setpoint should also be able to be eliviated
        """
        self.state_epo_reg_loss = epo_reg_loss
        self.state_task_loss = epo_task_loss
        if self.state_updater.update_setpoint():
            logger = Logger.get_logger(logger_name='main_out_logger', loglevel="INFO")
            self.setpoint4R = self.state_epo_reg_loss
            logger.info("!!!!!set point updated to {self.setpoint4R}!")


class FbOptSetpointControllerState():
    """
    abstract state pattern
    """
    def __init__(self):
        """
        """
        self.host = None

    def accept(self, controller):
        """
        set host for state
        """
        self.host = controller


class SliderAllComponent(FbOptSetpointControllerState):
    """
    concrete state pattern
    """
    def update_setpoint(self):
        """
        all components of R descreases regardless if ell decreases or not
        """
        if is_less_list_all(self.host.state_epo_reg_loss, self.host.setpoint4R):
            return True
        return False


class SliderAnyComponent(FbOptSetpointControllerState):
    """
    concrete state pattern
    """
    def update_setpoint(self):
        """
        if any component of R has decreased regardless if ell decreases
        """
        if is_less_list_any(self.host.state_epo_reg_loss, self.host.setpoint4R):
            self.host.transition_to(SliderAllComponent())
            return True
        return False


class DominateAnyComponent(FbOptSetpointControllerState):
    """
    concrete state pattern
    """
    def update_setpoint(self):
        """
        if any of the component of R loss has decreased together with ell loss
        """
        flag1 = is_less_list_any(self.host.state_epo_reg_loss, self.host.setpoint4R)
        flag2 = self.host.state_task_loss < self.host.setpoint4ell
        if flag2:
            self.host.setpoint4ell = self.host.state_task_loss
        return flag1 & flag2


class DominateAllComponent(FbOptSetpointControllerState):
    """
    concrete state pattern
    """
    def update_setpoint(self):
        """
        if each component of R loss has decreased and ell loss also decreased
        """
        flag1 = is_less_list_all(self.host.state_epo_reg_loss, self.host.setpoint4R)
        flag2 = self.host.state_task_loss < self.host.setpoint4ell
        if flag2:
            self.host.setpoint4ell = self.host.state_task_loss
        return flag1 & flag2
