"""
update hyper-parameters during training
"""
import numpy as np
from domainlab.utils.logger import Logger


def list_true(list1):
    """
    find out position of a list which has element True
    """
    arr_pos = np.arange(len(list1))[list1]
    return list(arr_pos)


def list_add(list1, list2):
    """
    add two lists
    """
    return [a + b for a, b in zip(list1, list2)]


def list_multiply(list1, coeff):
    """
    multiply a scalar to a list
    """
    return [ele * coeff for ele in list1]


def if_list_sign_agree(list1, list2):
    """
    each pair must have the same sign
    """
    list_agree = [a*b >= 0 for a, b in zip(list1, list2)]
    if not all(list_agree):
        raise RuntimeError(f"{list1} and {list2} can not be compared!")


def is_less_list_any(list1, list2):
    """
    judge if one list is less than the other
    """
    if_list_sign_agree(list1, list2)
    list_comparison = [a < b if a >= 0 and b >= 0 else a > b for a, b in zip(list1, list2)]
    return any(list_comparison), list_true(list_comparison)


def is_less_list_all(list1, list2):
    """
    judge if one list is less than the other
    """
    if_list_sign_agree(list1, list2)
    list_comparison = [a < b if a >= 0 and b >= 0 else a > b for a, b in zip(list1, list2)]
    return all(list_comparison)


def list_ma(list_state, list_input, coeff):
    """
    moving average of list
    """
    return [a * coeff + b * (1-coeff) for a, b in zip(list_state, list_input)]


class SetpointRewinder():
    """
    rewind setpoint if current loss exponential moving average is bigger than setpoint
    """
    def __init__(self, host):
        self.host = host
        self.counter = None
        self.epo_ma = None
        self.ref = None
        self.coeff_ma = 0.5
        self.setpoint_rewind = host.flag_setpoint_rewind

    def reset(self, epo_reg_loss):
        """
        when setpoint is adjusted
        """
        self.counter = 0
        self.epo_ma = [0.0 for _ in range(10)]  # FIXME
        self.ref = epo_reg_loss

    def observe(self, epo_reg_loss):
        """
        update moving average
        """
        if self.ref is None:
            self.reset(epo_reg_loss)
        self.epo_ma = list_ma(self.epo_ma, epo_reg_loss, self.coeff_ma)
        list_comparison_increase = [a < b for a, b in zip(self.ref, self.epo_ma)]
        list_comparison_above_setpoint = [a < b for a, b in zip(self.host.setpoint4R, self.epo_ma)]
        flag_increase = any(list_comparison_increase)
        flag_above_setpoint = any(list_comparison_above_setpoint)
        if flag_increase and flag_above_setpoint:
            self.counter += 1

        else:
            self.counter = 0
            self.reset(epo_reg_loss)

        if self.setpoint_rewind:
            if self.counter > 2 and self.counter <= 3:  # only allow self.counter = 2, 3 to rewind setpoing twice
                list_pos = list_true(list_comparison_above_setpoint)
                print(f"\n\n\n!!!!!!!setpoint too low at {list_pos}!\n\n\n")
                for pos in list_pos:
                    print(f"\n\n\n!!!!!!!rewinding setpoint at pos {pos} \
                        from {self.host.setpoint4R[pos]} to {self.epo_ma[pos]}!\n\n\n")
                    self.host.setpoint4R[pos] = self.epo_ma[pos]

            if self.counter > 3:
                self.host.transition_to(FixedSetpoint())
                self.counter = np.inf  # FIXME



class FbOptSetpointController():
    """
    update setpoint for mu
    """
    def __init__(self, state=None, args=None):
        """
        kwargs is a dictionary with key the hyper-parameter name and its value
        """
        if state is None:
            if args is not None and args.no_setpoint_update:
                state = FixedSetpoint()
            else:
                state = DominateAnyComponent()
        self.transition_to(state)
        self.flag_setpoint_rewind = args.setpoint_rewind == "yes"
        self.setpoint_rewinder = SetpointRewinder(self)
        self.state_task_loss = 0.0
        self.state_epo_reg_loss = [0.0 for _ in range(10)] # FIXME: 10 is the maximum number losses here
        self.coeff_ma_setpoint = args.coeff_ma_setpoint
        self.coeff_ma_output = args.coeff_ma_output_state
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

    def update_setpoint_ma(self, list_target, list_pos):
        """
        using moving average
        """
        target_ma = [self.coeff_ma_setpoint * a + (1 - self.coeff_ma_setpoint) * b
                     for a, b in zip(self.setpoint4R, list_target)]
        self.setpoint4R = [target_ma[i] if i in list_pos else self.setpoint4R[i] for i in range(len(target_ma))]

    def observe(self, epo_reg_loss, epo_task_loss):
        """
        read current epo_reg_loss continuously
        """
        self.state_epo_reg_loss = [self.coeff_ma_output*a + (1-self.coeff_ma_output)*b
                                   if a != 0.0 else b
                                   for a, b in zip(self.state_epo_reg_loss, epo_reg_loss)]
        if self.state_task_loss == 0.0:
            self.state_task_loss = epo_task_loss
        self.state_task_loss = self.coeff_ma_output * self.state_task_loss + \
            (1-self.coeff_ma_output) * epo_task_loss
        self.setpoint_rewinder.observe(self.state_epo_reg_loss)
        flag_update, list_pos = self.state_updater.update_setpoint()
        if flag_update:
            self.setpoint_rewinder.reset(self.state_epo_reg_loss)
            logger = Logger.get_logger(logger_name='main_out_logger', loglevel="INFO")
            logger.info(f"!!!!!set point old value {self.setpoint4R}!")
            self.update_setpoint_ma(self.state_epo_reg_loss, list_pos)
            logger.info(f"!!!!!set point updated to {self.setpoint4R}!")


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


class FixedSetpoint(FbOptSetpointControllerState):
    """
    do not update setpoint
    """
    def update_setpoint(self):
        """
        always return False so setpoint no update
        """
        return False, None


class SliderAllComponent(FbOptSetpointControllerState):
    """
    concrete state pattern
    """
    def update_setpoint(self):
        """
        all components of R descreases regardless if ell decreases or not
        """
        if is_less_list_all(self.host.state_epo_reg_loss, self.host.setpoint4R):
            return True, list(range(len(self.host.setpoint4R)))
        return False, None


class SliderAnyComponent(FbOptSetpointControllerState):
    """
    concrete state pattern
    """
    def update_setpoint(self):
        """
        if any component of R has decreased regardless if ell decreases
        """
        flag, list_pos = is_less_list_any(self.host.state_epo_reg_loss, self.host.setpoint4R)
        if flag:
            self.host.transition_to(SliderAllComponent())
            return True, list_pos
        return False, list_pos


class DominateAnyComponent(FbOptSetpointControllerState):
    """
    concrete state pattern
    """
    def update_setpoint(self):
        """
        if any of the component of R loss has decreased together with ell loss
        """
        flag1, list_pos = is_less_list_any(self.host.state_epo_reg_loss, self.host.setpoint4R)
        flag2 = self.host.state_task_loss < self.host.setpoint4ell
        if flag2:
            self.host.setpoint4ell = self.host.state_task_loss
        return flag1 & flag2, list_pos


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
        return flag1 & flag2, list(range(len(self.host.setpoint4R)))