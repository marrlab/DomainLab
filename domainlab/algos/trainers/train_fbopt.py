"""
feedback optimization
"""
import copy

from domainlab.algos.trainers.a_trainer import AbstractTrainer
from domainlab.algos.trainers.train_basic import TrainerBasic
from domainlab.algos.trainers.fbopt import HyperSchedulerFeedback


class HyperSetter():
    """
    mock object to force hyper-parameter in the model
    """
    def __init__(self, dict_hyper):
        self.dict_hyper = dict_hyper

    def __call__(self, epoch=None):
        return self.dict_hyper


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
        # model.hyper_init will register the hyper-parameters of the model to scheduler
        self.hyper_scheduler = self.model.hyper_init(scheduler, trainer=self)

    def before_tr(self):
        """
        before training begins, construct helper objects
        """
        self.set_scheduler(scheduler=HyperSchedulerFeedback)
        self.model.evaluate(self.loader_te, self.device)
        self.inner_trainer = TrainerBasic()  # look ahead
        # here we need a mechanism to generate deep copy of the model
        self.inner_trainer.init_business(
            copy.deepcopy(self.model), self.task, self.observer, self.device, self.aconf,
            flag_accept=False)

    def opt_theta(self, dict4mu, dict_theta0):
        """
        operator for theta, move gradient for one epoch, then check if criteria is met
        this method will be invoked by the hyper-parameter scheduling object
        """
        self.inner_trainer.model.set_params(dict_theta0)
        # mock the model hyper-parameter to be from dict4mu
        self.inner_trainer.model.hyper_update(epoch=None, fun_scheduler=HyperSetter(dict4mu))
        # hide implementation details of inner_trainer
        for _, (tensor_x, vec_y, vec_d, *others) in enumerate(self.inner_trainer.loader_tr):
            self.inner_trainer.train_batch(tensor_x, vec_y, vec_d, others)  # update inner_net
        dict_par = self.inner_trainer.model.name_parameters()
        return dict_par

    def eval_loss(self, dict4mu, dict_theta):
        """
        evaluate the penalty function value
        """
        temp_model = copy.deepcopy(self.model)
        # mock the model hyper-parameter to be from dict4mu
        temp_model.hyper_update(epoch=None, fun_scheduler=HyperSetter(dict4mu))
        temp_model.set_params(dict_theta)
        epo_reg_loss = 0
        epo_task_loss = 0
        epo_p_loss = 0  # penalized loss
        # FIXME: will loader be corupted? if called at different places? if we do not make deep copy
        for _, (tensor_x, vec_y, vec_d, *_) in enumerate(self.loader_tr):
            b_reg_loss = temp_model.cal_reg_loss(tensor_x, vec_y, vec_d).sum()
            b_task_loss = temp_model.cal_task_loss(tensor_x, vec_y, vec_d).sum()
            # sum will kill the dimension of the mini batch
            b_p_loss = temp_model.cal_p_loss(tensor_x, vec_y).sum()
            epo_reg_loss += b_reg_loss
            epo_task_loss += b_task_loss
            epo_p_loss += b_p_loss
        return epo_p_loss

    def tr_epoch(self, epoch):
        self.model.train()
        self.hyper_scheduler.search_mu(
            dict(self.model.named_parameters()))   # if mu not found, will terminate
        self.model.set_params(self.hyper_scheduler.dict_theta)
        flag_stop = self.observer.update(epoch)  # notify observer
        return flag_stop
