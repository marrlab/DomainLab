"""
update hyper-parameters during training
"""
import warnings
import numpy as np
from domainlab.algos.trainers.train_basic import TrainerBasic
from domainlab.utils.logger import Logger


class TrainerHyperScheduler(TrainerBasic):
    """
    TrainerHyperScheduler
    """
    def set_scheduler(self, scheduler, total_steps,
                      flag_update_epoch=False,
                      flag_update_batch=False):
        """
        set the warmup strategy from objective scheduler
        set wheter the hyper-parameter scheduling happens per epoch or per batch
        """
        self.hyper_scheduler = self.model.hyper_init(scheduler)
        self.flag_update_hyper_per_epoch = flag_update_epoch
        self.flag_update_hyper_per_batch = flag_update_batch
        self.hyper_scheduler.set_steps(total_steps=total_steps)

    def after_batch(self, epoch, ind_batch):
        """
        if hyper-parameters should be updated per batch, then step
        should be set to epoch*self.num_batches + ind_batch
        """
        if self.flag_update_hyper_per_batch:
            self.model.hyper_update(epoch*self.num_batches + ind_batch, self.hyper_scheduler)
        return super().after_batch(epoch, ind_batch)

    def before_tr(self):
        if self.hyper_scheduler is None:
            logger = Logger.get_logger()
            logger.warning("hyper-parameter scheduler not set,"
                           "going to use default Warmpup and epoch update")
            warnings.warn("hyper-parameter scheduler not set, "
                          "going to use default Warmpup and epoch update")
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
        for key, val_setpoint in self.dict_par_setpoint.items():
            dict_rst[key] = self.warmup(val_setpoint, epoch)
        return dict_rst


class HyperSchedulerWarmupExponential(HyperSchedulerWarmup):
    """
    HyperScheduler Exponential
    """
    def aneal(self, par_setpoint, epoch):
        """
        start from a small value of par to ramp up the steady state value using
        number of total_steps
        :param epoch:
        """
        ratio = ((epoch+1) * 1.) / self.total_steps
        denominator = (1. + np.exp(-10 * ratio))
        # ratio is 0, denom is 2, 2/denom is 1, return is 0
        # ratio is 1, denom is 1+exp(-10), 2/denom is 2/(1+exp(-10))=2, return is 1
        # exp(-10)=4.5e-5 is approximately 0
        # slowly increase the regularization weight from 0 to 1*alpha as epochs goes on
        return float((2. / denominator - 1) * par_setpoint)
