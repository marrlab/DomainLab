"""
update hyper-parameters during training
"""
import warnings
import numpy as np
from domainlab.algos.trainers.train_basic import TrainerBasic


class TrainerVisitor(TrainerBasic):
    """
    TrainerVisitor
    """
    def set_scheduler(self, scheduler, total_steps,
                      flag_update_epoch=False,
                      flag_update_batch=False):
        """
        set the warmup or anealing strategy
        """
        self.hyper_scheduler = self.model.hyper_init(scheduler)
        self.flag_update_hyper_per_epoch = flag_update_epoch
        self.flag_update_hyper_per_batch = flag_update_batch
        self.hyper_scheduler.set_steps(total_steps=total_steps)

    def after_batch(self, epoch, ind_batch):
        if self.flag_update_hyper_per_batch:
            self.model.hyper_update(epoch, self.hyper_scheduler)
        return super().after_batch(epoch, ind_batch)

    def before_tr(self):
        if self.hyper_scheduler is None:
            warnings.warn("hyper-parameter scheduler not set, \
                          going to use default Warmpup and epoch update")
            self.set_scheduler(HyperSchedulerWarmup,
                               total_steps=self.aconf.warmup,
                               flag_update_epoch=True)

    def tr_epoch(self, epoch):
        """
        update hyper-parameters only per epoch
        """
        if self.flag_update_hyper_per_epoch:
            self.model.hyper_update(epoch, self.hyper_scheduler)
        return super().tr_epoch(epoch)

    def tr_batch(self, epoch, ind_batch):
        """
        anneal hyper-parameter for each batch
        """
        self.model.hyper_update(epoch*self.num_batches + ind_batch, self.hyper_scheduler)
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
        for key, _ in self.dict_par_setpoint.items():
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
