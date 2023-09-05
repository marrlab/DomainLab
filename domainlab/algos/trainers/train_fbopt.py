"""
feedback optimization
"""
import copy

from domainlab.algos.trainers.a_trainer import AbstractTrainer
from domainlab.algos.trainers.train_basic import TrainerBasic
from domainlab.algos.trainers.fbopt import HyperSchedulerFeedback


class TrainerFbOpt(AbstractTrainer):
    """
    feedback optimization
    """
    def set_scheduler(self, scheduler=HyperSchedulerFeedback):
        """
        Args:
            scheduler: The class name of the scheduler, the object corresponding to
            this class name will be created inside model
        """
        self.hyper_scheduler = self.model.hyper_init(scheduler)

    def before_tr(self):
        """
        check the performance of randomly initialized weight
        """
        self.set_scheduler()
        self.model.evaluate(self.loader_te, self.device)
        self.inner_trainer = TrainerBasic()
        self.inner_trainer.init_business(
            self.model, self.task, self.observer, self.device, self.aconf,
            flag_accept=False)

    def opt_theta(self, mmu):
        """
        operator for theta, move gradient for a few steps, then check if criteria is met
        """

    def tr_epoch(self, epoch):
        self.model.train()
        self.hyper_scheduler.search_mu()

        for ind_batch, (tensor_x, vec_y, vec_d, *others) in enumerate(self.loader_tr):
            inner_net = copy.deepcopy(self.model)
            self.inner_trainer.model = inner_net   # FORCE replace model
            self.inner_trainer.train_batch(
                tensor_x, vec_y, vec_d, others)  # update inner_net

            loss_look_forward = inner_net.cal_task_loss(tensor_x, vec_y)
            loss_source = self.model.cal_loss(tensor_x, vec_y, vec_d, others)
            loss = loss_source.sum() + self.aconf.gamma_reg * loss_look_forward.sum()
            loss.backward()
            self.optimizer.step()
            self.after_batch(epoch, ind_batch)
        flag_stop = self.observer.update(epoch)  # notify observer
        return flag_stop
