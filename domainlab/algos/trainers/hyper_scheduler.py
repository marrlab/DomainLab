"""
update hyper-parameters during training
"""
import numpy as np


class HyperSchedulerWarmupLinear():
    """
    HyperSchedulerWarmupLinear
    """
    def __init__(self, trainer, **kwargs):
        """
        kwargs is a dictionary with key the hyper-parameter name and its value
        """
        self.trainer = trainer
        self.dict_par_setpoint = kwargs
        self.total_steps = None

    def set_steps(self, total_steps):
        """
        set number of total_steps to gradually change optimization parameter
        """
        self.total_steps = total_steps

    def warmup(self, par_setpoint, epoch):
        """warmup.
        start from a small value of par to ramp up the steady state value using
        # total_steps
        :param epoch:
        """
        ratio = ((epoch+1) * 1.) / self.total_steps
        list_par = [par_setpoint, par_setpoint * ratio]
        par = min(list_par)
        return par

    def __call__(self, epoch):
        dict_rst = {}
        for key, val_setpoint in self.dict_par_setpoint.items():
            dict_rst[key] = self.warmup(val_setpoint, epoch)
        return dict_rst


class HyperSchedulerWarmupExponential(HyperSchedulerWarmupLinear):
    """
    HyperScheduler Exponential
    """
    def warmup(self, par_setpoint, epoch):
        """
        start from a small value of par to ramp up the steady state value using
        number of total_steps
        :param epoch:
        """
        percent_steps = ((epoch+1) * 1.) / self.total_steps
        denominator = 1. + np.exp(-10 * percent_steps)
        ratio = (2. / denominator - 1)
        # percent_steps is 0, denom is 2, 2/denom is 1, ratio is 0
        # percent_steps is 1, denom is 1+exp(-10), 2/denom is 2/(1+exp(-10))=2, ratio is 1
        # exp(-10)=4.5e-5 is approximately 0
        # slowly increase the regularization weight from 0 to 1*alpha as epochs goes on
        parval = float(ratio * par_setpoint)
        return parval
