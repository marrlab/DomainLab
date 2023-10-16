"""
update hyper-parameters during training
"""
import os

from torch.utils.tensorboard import SummaryWriter

import numpy as np
from domainlab.utils.logger import Logger
from domainlab.algos.trainers.fbopt_setpoint_ada import FbOptSetpointController
from domainlab.algos.trainers.fbopt_setpoint_ada import if_list_sign_agree


class StubSummaryWriter():
    """
    # stub writer for tensorboard that ignores all messages
    """

    def add_scalar(self, *args, **kwargs):
        """
        stub, pass do nothing
        """

    def add_scalars(self, *args, **kwargs):
        """
        stub, pass, do nothing
        """


class HyperSchedulerFeedback():
    """
    design $\\mu$$ sequence based on state of penalized loss
    """
    def __init__(self, trainer, **kwargs):
        """
        kwargs is a dictionary with key the hyper-parameter name and its value
        """
        self.trainer = trainer
        self.init_mu = trainer.aconf.mu_init
        self.mu_min = trainer.aconf.mu_min
        self.mu_clip = trainer.aconf.mu_clip

        self.mmu = kwargs
        # force initial value of mu
        self.mmu = {key: self.init_mu for key, val in self.mmu.items()}

        self.set_point_controller = FbOptSetpointController(
            args=self.trainer.aconf)

        self.k_i_control = trainer.aconf.k_i_gain
        self.overshoot_rewind = trainer.aconf.overshoot_rewind == "yes"
        self.delta_epsilon_r = None
        # NOTE: this value will be set according to initial evaluation of
        # neural network
        self.activation_clip = trainer.aconf.exp_shoulder_clip
        if trainer.aconf.no_tensorboard:
            self.writer = StubSummaryWriter()
        else:
            str_job_id = os.environ.get('SLURM_JOB_ID', '')
            self.writer = SummaryWriter(comment=str_job_id)
        self.coeff_ma = trainer.aconf.coeff_ma

    def get_setpoing4r(self):
        """
        get setpoint list
        """
        return self.set_point_controller.setpoint4R

    def set_setpoint(self, list_setpoint4r, setpoint4ell):
        """
        set the setpoint
        """
        self.set_point_controller.setpoint4R = list_setpoint4r
        self.set_point_controller.setpoint4ell = setpoint4ell

    def cal_delta4control(self, list1, list_setpoint):
        """
        list difference
        """
        if_list_sign_agree(list1, list_setpoint)
        delta_epsilon_r = [a - b for a, b in zip(list1, list_setpoint)]
        if self.delta_epsilon_r is None:
            self.delta_epsilon_r = delta_epsilon_r
        else:
            # PI control.
            # self.delta_epsilon_r is the previous time step.
            # delta_epsilon_r is the current time step
            self.delta_epsilon_r = self.cal_delta_integration(
                self.delta_epsilon_r, delta_epsilon_r, self.coeff_ma)

    def cal_delta_integration(self, list_old, list_new, coeff):
        """
        ma of delta
        """
        return [(1 - coeff) * a + coeff * b
                for a, b in zip(list_old, list_new)]

    def tackle_overshoot(self,
                         activation, epo_reg_loss, list_str_multiplier_na):
        """
        tackle overshoot
        """
        list_overshoot = [i if (a - b) * (self.delta_epsilon_r[i]) < 0
                          else None
                          for i, (a, b) in
                          enumerate(
                              zip(epo_reg_loss,
                                  self.set_point_controller.setpoint4R))]
        for ind in list_overshoot:
            if ind is not None:
                logger = Logger.get_logger(
                    logger_name='main_out_logger', loglevel="INFO")
                logger.info(f"delta integration: {self.delta_epsilon_r}")
                logger.info(f"overshooting at  pos \
                            {ind} of activation: {activation}")
                logger.info(f"name reg loss:{list_str_multiplier_na}")
                if self.overshoot_rewind:
                    activation[ind] = 0.0
                    logger.info(f"PID controller set to zero now, \
                                new activation: {activation}")
        return activation

    def cal_activation(self):
        """
        calculate activation on exponential shoulder
        """
        setpoint = self.get_setpoing4r()
        activation = [self.k_i_control * val if setpoint[i] > 0
                      else self.k_i_control * (-val) for i, val
                      in enumerate(self.delta_epsilon_r)]
        if self.activation_clip is not None:
            activation = [np.clip(val,
                                  a_min=-1 * self.activation_clip,
                                  a_max=self.activation_clip)
                          for val in activation]
        return activation

    def search_mu(self, epo_reg_loss, epo_task_loss, epo_loss_tr,
                  list_str_multiplier_na, miter):
        """
        start from parameter dictionary dict_theta: {"layer":tensor},
        enlarge mu w.r.t. its current value
        to see if the criteria is met
        $$\\mu^{k+1}=mu^{k}exp(rate_mu*[R(\\theta^{k})-ref_R])$$
        """
        logger = Logger.get_logger(
            logger_name='main_out_logger', loglevel="INFO")
        logger.info(f"before controller: current mu: {self.mmu}")
        logger.info(f"epo reg loss: {epo_reg_loss}")
        logger.info(f"name reg loss:{list_str_multiplier_na}")
        self.cal_delta4control(epo_reg_loss, self.get_setpoing4r())
        activation = self.cal_activation()
        # overshoot handling
        activation = self.tackle_overshoot(
            activation, epo_reg_loss, list_str_multiplier_na)
        list_gain = np.exp(activation)
        dict_gain = dict(zip(list_str_multiplier_na, list_gain))
        target = self.dict_multiply(self.mmu, dict_gain)
        self.mmu = self.dict_clip(target)
        logger = Logger.get_logger(
            logger_name='main_out_logger', loglevel="INFO")
        logger.info(f"after contoller: current mu: {self.mmu}")

        for key, val in self.mmu.items():
            self.writer.add_scalar(f'dyn_mu/{key}', val, miter)
            self.writer.add_scalar(
                f'controller_gain/{key}', dict_gain[key], miter)
            ind = list_str_multiplier_na.index(key)
            self.writer.add_scalar(
                f'delta/{key}', self.delta_epsilon_r[ind], miter)
        for i, (reg_dyn, reg_set) in \
                enumerate(zip(epo_reg_loss, self.get_setpoing4r())):
            self.writer.add_scalar(
                f'lossrd/dyn_{list_str_multiplier_na[i]}', reg_dyn, miter)
            self.writer.add_scalar(
                f'lossrs/setpoint_{list_str_multiplier_na[i]}', reg_set, miter)

            self.writer.add_scalars(
                f'loss_rds/loss_{list_str_multiplier_na[i]} with setpoint',
                {f'lossr/loss_{list_str_multiplier_na[i]}': reg_dyn,
                 f'lossr/setpoint_{list_str_multiplier_na[i]}': reg_set,
                 }, miter)
            self.writer.add_scalar(
                f'x-axis=loss_ell task vs y-axis=loss \
                r/dyn{list_str_multiplier_na[i]}',
                reg_dyn, epo_task_loss)
        self.writer.add_scalar('loss_task/penalized', epo_loss_tr, miter)
        self.writer.add_scalar('loss_task/ell', epo_task_loss, miter)
        acc_te = 0
        acc_val = 0
        acc_sel = 0
        acc_set = 0

        if miter > 1:
            acc_te = self.trainer.observer.metric_te["acc"]
            acc_val = self.trainer.observer.metric_val["acc"]
            acc_sel = self.trainer.observer.model_sel.sel_model_te_acc
            acc_set = \
                self.trainer.observer.model_sel.oracle_last_setpoint_sel_te_acc
        self.writer.add_scalar("acc/te", acc_te, miter)
        self.writer.add_scalar("acc/val", acc_val, miter)
        self.writer.add_scalar("acc/sel", acc_sel, miter)
        self.writer.add_scalar("acc/setpoint", acc_set, miter)

    def dict_clip(self, dict_base):
        """
        clip each entry of the mu according to pre-set self.mu_clip
        """
        return {key: np.clip(val, a_min=self.mu_min, a_max=self.mu_clip)
                for key, val in dict_base.items()}

    def dict_is_zero(self, dict_mu):
        """
        check if hyper-parameter start from zero
        """
        for key in dict_mu.keys():
            if dict_mu[key] == 0.0:
                return True
        return False

    def dict_multiply(self, dict_base, dict_multiplier):
        """
        multiply a float to each element of a dictionary
        """
        return {
            key: val * dict_multiplier[key] for key, val in dict_base.items()}

    def update_setpoint(self, epo_reg_loss, epo_task_loss):
        """
        update setpoint
        """
        return self.set_point_controller.observe(epo_reg_loss, epo_task_loss)
