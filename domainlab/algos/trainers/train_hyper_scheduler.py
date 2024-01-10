"""
update hyper-parameters during training
"""
from domainlab.algos.trainers.train_basic import TrainerBasic
from domainlab.algos.trainers.hyper_scheduler import HyperSchedulerWarmupLinear
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

        Args:
            scheduler: The class name of the scheduler, the object corresponding to
            this class name will be created inside model
            total_steps: number of steps to change the hyper-parameters
            flag_update_epoch: if hyper-parameters should be changed per epoch
            flag_update_batch: if hyper-parameters should be changed per batch
        """
        self.hyper_scheduler = self.model.hyper_init(scheduler)
        # let model register its hyper-parameters to the scheduler
        self.flag_update_hyper_per_epoch = flag_update_epoch
        self.flag_update_hyper_per_batch = flag_update_batch
        self.hyper_scheduler.set_steps(total_steps=total_steps)

    def before_batch(self, epoch, ind_batch):
        """
        if hyper-parameters should be updated per batch, then step
        should be set to epoch*self.num_batches + ind_batch
        """
        if self.flag_update_hyper_per_batch:
            self.model.hyper_update(epoch*self.num_batches + ind_batch, self.hyper_scheduler)
        return super().before_batch(epoch, ind_batch)

    def before_tr(self):
        if self.hyper_scheduler is None:
            logger = Logger.get_logger()
            logger.warning("hyper-parameter scheduler not set,"
                           "going to use default Warmpup and epoch update")
            self.set_scheduler(HyperSchedulerWarmupLinear,
                               total_steps=self.aconf.warmup,
                               flag_update_epoch=True)

    def tr_epoch(self, epoch):
        """
        update hyper-parameters only per epoch
        """
        if self.flag_update_hyper_per_epoch:
            self.model.hyper_update(epoch, self.hyper_scheduler)
        return super().tr_epoch(epoch)
