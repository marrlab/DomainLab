import warnings
import numpy as np

from domainlab.algos.trainers.train_basic import TrainerBasic


class TrainerVisitor(TrainerBasic):
    """
    TrainerVisitor
    """
    def set_scheduler(self, scheduler):
        """
        set the warmup or anealing strategy
        """
        self.hyper_scheduler = self.model.hyper_init(scheduler)

    def before_tr(self):
        if self.hyper_scheduler is None:
            warnings.warn("hyper-parameter scheduler not set, going to use default WarmpUP")
            self.hyper_scheduler = self.model.hyper_init(HyperSchedulerWarmup)
        # @FIXME: is there a way to make this more general?
        self.hyper_scheduler.set_steps(total_steps=self.aconf.warmup)

    def tr_epoch(self, epoch):
        self.model.hyper_update(epoch, self.hyper_scheduler)
        return super().tr_epoch(epoch)


class HyperSchedulerWarmup():
    """
    HyperSchedulerWarmup
    """
    def __init__(self, **kwargs):
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
        for key in self.dict_par_setpoint:
            dict_rst[key] = self.warmup(self.dict_par_setpoint[key], epoch)
        return dict_rst


class HyperSchedulerAneal(HyperSchedulerWarmup):
    """
    HyperSchedulerAneal
    """
    def aneal(self, epoch, alpha):
        """
        start from a small value of par to ramp up the steady state value using
        number of total_steps
        :param epoch:
        """
        ratio = ((epoch+1) * 1.) / self.total_steps
        denominator = (1. + np.exp(-10 * ratio))
        return float((2. / denominator - 1) * alpha)

    def __call__(self, epoch):
        dict_rst = {}
        for key, val in self.dict_par_setpoint.items():
            dict_rst[key] = self.aneal(epoch, val)
        return dict_rst
