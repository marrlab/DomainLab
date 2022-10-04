import numpy as np

from domainlab.algos.trainers.train_basic import TrainerBasic


class TrainerVisitor(TrainerBasic):
    def before_tr(self):
        self.hyper_scheduler = self.model.hyper_init(HyperSchedulerWarmup)
        self.hyper_scheduler.set_steps(steps=self.aconf.warmup)  # FIXME: is there a way to make this more general?

    def tr_epoch(self, epoch):
        self.model.hyper_update(epoch, self.hyper_scheduler)
        return super().tr_epoch(epoch)


class HyperSchedulerWarmup():
    def __init__(self, **kwargs):
        self.dict_par_setpoint = kwargs
        self.steps = None

    def set_steps(self, steps):
        self.steps = steps

    def warmup(self, par_setpoint, epoch):
        """warmup.
        start from a small value of par to ramp up the steady state value using
        # steps
        :param epoch:
        """
        ratio = ((epoch+1) * 1.) / self.steps
        list_par = [par_setpoint, par_setpoint * ratio]
        par = min(list_par)
        return par

    def __call__(self, epoch):
        dict_rst = {}
        for key in self.dict_par_setpoint:
            dict_rst[key] = self.warmup(self.dict_par_setpoint[key], epoch)
        return dict_rst


class HyperSchedulerAneal(HyperSchedulerWarmup):
    def aneal(self, epoch):
        """warmup.
        start from a small value of par to ramp up the steady state value using
        # steps
        :param epoch:
        """
        ratio = ((epoch+1) * 1.) / self.steps
        return float((2. / (1. + np.exp(-10 * ratio)) - 1) * self._alpha)

    def __call__(self, epoch):
        dict_rst = {}
        for key in self.dict_par_setpoint:
            dict_rst[key] = self.aneal(epoch)
        return dict_rst
