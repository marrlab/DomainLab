"""
update hyper-parameters during training
"""
import copy
import os

from torch.utils.tensorboard import SummaryWriter

import numpy as np
from domainlab.utils.logger import Logger
from domainlab.algos.trainers.fbopt_setpoint_ada import FbOptSetpointController


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


class HyperSchedulerFeedbackAlternave():
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
        self.mmu = kwargs
        self.mmu = {key: self.init_mu for key, val in self.mmu.items()}
        self.ploss_old_theta_old_mu = None
        self.ploss_old_theta_new_mu = None
        self.ploss_new_theta_old_mu = None
        self.ploss_new_theta_new_mu = None
        self.delta_mu = trainer.aconf.delta_mu
        # for exponential increase of mu, mu can not be starting from zero
        self.beta_mu = trainer.aconf.beta_mu
        self.dict_theta_bar = None
        self.dict_theta = copy.deepcopy(dict(self.trainer.model.named_parameters()))
        # theta_ref should be equal to either theta or theta bar as reference
        # since theta_ref will be used to judge if criteria is met
        self.dict_theta_ref = None
        self.budget_mu_per_step = trainer.aconf.budget_mu_per_step
        self.budget_theta_update_per_mu = trainer.aconf.budget_theta_update_per_mu
        self.count_found_operator = 0
        self.count_search_mu = 0
        ########################################
        self.set_point_controller = FbOptSetpointController(args=self.trainer.aconf)
        self.k_i_control = trainer.aconf.k_i_gain
        self.overshoot_rewind = trainer.aconf.overshoot_rewind == "yes"
        self.delta_epsilon_r = False  # False here just used to decide if value first use or not
        # NOTE: this value will be set according to initial evaluation of neural network
        self.mu_clip = trainer.aconf.mu_clip
        self.activation_clip = trainer.aconf.exp_shoulder_clip
        if trainer.aconf.no_tensorboard:
            self.writer = StubSummaryWriter()
        else:
            str_job_id = os.environ.get('SLURM_JOB_ID', '')
            self.writer = SummaryWriter(comment=str_job_id)
        self.coeff_ma = trainer.aconf.coeff_ma
        self.epsilon_r = False

    def get_setpoint4R(self):
        """
        get setpoint list
        """
        return self.set_point_controller.setpoint4R

    def set_setpoint(self, list_setpoint4R, setpoint4ell):
        """
        set the setpoint
        """
        self.set_point_controller.setpoint4R = list_setpoint4R
        self.set_point_controller.setpoint4ell = setpoint4ell

    def update_anchor(self, dict_par):
        """
        update the last ensured value of theta^{(k)}
        """
        self.dict_theta = copy.deepcopy(dict_par)

    def set_theta_ref(self):
        """
        # theta_ref should be equal to either theta or theta bar as reference
        # since theta_ref will be used to judge if criteria is met
        """
        if self.trainer.aconf.anchor_bar:
            self.dict_theta_ref = copy.deepcopy(self.dict_theta_bar)
        else:
            self.dict_theta_ref = copy.deepcopy(self.dict_theta)

    def cal_delta4control(self, list1, list_setpoint):
        """
        list difference
        """
        return [a - b for a, b in zip(list1, list_setpoint)]

    def cal_delta_integration(self, list_old, list_new, coeff):
        """
        ma of delta
        """
        return [(1-coeff)*a + coeff*b for a, b in zip(list_old, list_new)]

    def search_mu(self, epo_reg_loss, epo_task_loss, epo_loss_tr, list_str_multiplier_na, dict_theta=None, miter=None):
        """
        start from parameter dictionary dict_theta: {"layer":tensor},
        enlarge mu w.r.t. its current value
        to see if the criteria is met
        $$\\mu^{k+1}=mu^{k}exp(rate_mu*[R(\\theta^{k})-epsilon_R])$$
        """
        delta_epsilon_r = self.cal_delta4control(epo_reg_loss, self.get_setpoint4R())
        # TODO: can be replaced by a controller
        if self.delta_epsilon_r is False:
            self.delta_epsilon_r = delta_epsilon_r
        else:
            # PI control.
            # self.delta_epsilon_r is the previous time step.
            # delta_epsilon_r is the current time step
            self.delta_epsilon_r = self.cal_delta_integration(
                self.delta_epsilon_r, delta_epsilon_r, self.coeff_ma)
        # FIXME: here we can not sum up delta_epsilon_r directly, but normalization also makes no sense, the only way is to let gain as a dictionary
        activation = [self.k_i_control * val for val in self.delta_epsilon_r]
        if self.activation_clip is not None:
            activation = [np.clip(val, a_min=-1 * self.activation_clip, a_max=self.activation_clip)
                          for val in activation]
        # overshoot handling
        list_overshoot = [i if a < b and self.delta_epsilon_r[i] > b else None for i, (a, b) in enumerate(zip(epo_reg_loss, self.set_point_controller.setpoint4R))]
        for ind in list_overshoot:
            if ind is not None:
                logger = Logger.get_logger(logger_name='main_out_logger', loglevel="INFO")
                logger.info(f"overshooting at  pos {ind} of {activation}")
                if self.overshoot_rewind:
                    activation[ind] = 0.0
                    logger.info(f"PID controller set to zero now {activation}")
        list_gain = np.exp(activation)
        target = self.dict_multiply(self.mmu, list_gain)
        self.mmu = self.dict_clip(target)

        for key, val in self.mmu.items():
            self.writer.add_scalar(f'mmu/{key}', val, miter)
        for i, (reg_dyn, reg_set) in enumerate(zip(epo_reg_loss, self.get_setpoint4R())):
            self.writer.add_scalar(f'regd/dyn_{list_str_multiplier_na[i]}', reg_dyn, miter)
            self.writer.add_scalar(f'regs/setpoint_{list_str_multiplier_na[i]}', reg_set, miter)

            self.writer.add_scalars(
                f'regds/dyn_{list_str_multiplier_na[i]} with setpoint',
                {f'reg/dyn_{list_str_multiplier_na[i]}': reg_dyn,
                 f'reg/setpoint_{list_str_multiplier_na[i]}': reg_set,
                 }, miter)
            self.writer.add_scalar(
                f'x-axis=task vs y-axis=reg/dyn{list_str_multiplier_na[i]}', reg_dyn, epo_task_loss)
        self.writer.add_scalar('loss_penalized', epo_loss_tr, miter)
        self.writer.add_scalar('task', epo_task_loss, miter)
        acc_te = 0
        acc_val = 0

        if miter > 1:
            acc_te = self.trainer.observer.metric_te["acc"]
            acc_val = self.trainer.observer.metric_val["acc"]
        self.writer.add_scalar("acc/te", acc_te, miter)
        self.writer.add_scalar("acc/val", acc_val, miter)

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

    def dict_multiply(self, dict_base, list_multiplier):
        """
        multiply a float to each element of a dictionary
        """
        list_keys = list(dict_base.keys())
        list_zip = zip(list_keys, list_multiplier)
        dict_multiplier = dict(list_zip)
        # NOTE: allow multipler be bigger than 1
        return {key: val*dict_multiplier[key] for key, val in dict_base.items()}

    def update_setpoint(self, epo_reg_loss, epo_task_loss):
        """
        update setpoint
        """
        self.set_point_controller.observe(epo_reg_loss, epo_task_loss)
