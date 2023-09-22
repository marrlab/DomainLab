"""
update hyper-parameters during training
"""
import copy
import torch
import numpy as np
from domainlab.utils.logger import Logger
from torch.utils.tensorboard import SummaryWriter


class HyperSchedulerFeedbackAlternave():
    """
    design $\\mu$$ sequence based on state of penalized loss
    """
    def __init__(self, trainer, **kwargs):
        """
        kwargs is a dictionary with key the hyper-parameter name and its value
        """
        self.trainer = trainer
        self.mmu = kwargs
        self.mmu = {key: 1.0 for key, val in self.mmu.items()}
        self.ploss_old_theta_old_mu = None
        self.ploss_old_theta_new_mu = None
        self.ploss_new_theta_old_mu = None
        self.ploss_new_theta_new_mu = None
        self.delta_mu = trainer.aconf.delta_mu
        self.init_mu = trainer.aconf.init_mu4beta
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
        # FIXME: make the following a vector, (or  dictionary)
        self.k_p_control = 0.001
        self.delta_epsilon_r = False  # False here just used to decide if value first use or not
        self.reg_lower_bound_as_setpoint = None
        # NOTE: this value will be set according to initial evaluation of neural network
        self.mu_clip = 10000
        self.activation_clip = 10  # untested dummy value
        self.writer = SummaryWriter()
        self.coeff_ma = 0.5
        self.epsilon_r = False

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
        return [a - b for a, b in zip(list1, list_setpoint)]

    def cal_delta_integration(self, list_old, list_new, coeff):
        return [(1-coeff)*a + coeff*b for a, b in zip(list_old, list_new)]

    def search_mu(self, dict_theta=None, miter=None):
        """
        start from parameter dictionary dict_theta: {"layer":tensor},
        enlarge mu w.r.t. its current value
        to see if the criteria is met
        $$\\mu^{k+1}=mu^{k}exp(rate_mu*[R(\\theta^{k})-epsilon_R])$$
        """
        epo_reg_loss, epos_task_loss = self.trainer.eval_r_loss()
        # FIXME: use dictionary to replace scalar representation
        # delta_epsilon_r = epo_reg_loss - self.reg_lower_bound_as_setpoint
        delta_epsilon_r = self.cal_delta4control(epo_reg_loss, self.reg_lower_bound_as_setpoint)
        # TODO: can be replaced by a controller
        if self.delta_epsilon_r is False:
            self.delta_epsilon_r = delta_epsilon_r
        else:
            # PI control.
            # self.delta_epsilon_r is the previous time step.
            # delta_epsilon_r is the current time step
            # self.delta_epsilon_r = (1 - self.coeff_ma) * self.delta_epsilon_r + self.coeff_ma * delta_epsilon_r
            self.delta_epsilon_r = self.cal_delta_integration(self.delta_epsilon_r, delta_epsilon_r, self.coeff_ma)
        # FIXME: here we can not sum up selta_epsilon_r directly, but normalization also makes no sense, the only way is to let gain as a dictionary
        activation = [self.k_p_control * val for val in self.delta_epsilon_r]
        if self.activation_clip is not None:
            activation = [np.clip(val, a_min=-1 * self.activation_clip, a_max=self.activation_clip)
                          for val in activation]
        list_gain = np.exp(activation)
        target = self.dict_multiply(self.mmu, list_gain)
        self.mmu = self.dict_clip(target)
        val = list(self.mmu.values())[0]
        self.writer.add_scalar('mmu', val, miter)

        for i, (reg_dyn, reg_set) in enumerate(zip(epo_reg_loss, self.reg_lower_bound_as_setpoint)):
            self.writer.add_scalar(f'reg/dyn{i}', reg_dyn, miter)
            self.writer.add_scalar(f'reg/setpoint{i}', reg_set, miter)

            self.writer.add_scalars(f'reg/dyn{i} & reg/setpoint{i}', {
                f'reg/dyn{i}': reg_dyn,
                f'reg/setpoint{i}': reg_set,
            }, miter)
            self.writer.add_scalar(f'x-axis=task vs y-axis=reg/dyn{i}', reg_dyn, epos_task_loss)

        #self.writer.add_scalar('loss_penalized', epos_task_loss+, miter)
        self.writer.add_scalar(f'task', epos_task_loss, miter)
        self.dict_theta = self.trainer.opt_theta(self.mmu, dict(self.trainer.model.named_parameters()))
        return True

    def dict_clip(self, dict_base):
        """
        clip each entry of the mu according to pre-set self.mu_clip
        """
        return {key: np.clip(val, a_min=0.0, a_max=self.mu_clip) for key, val in dict_base.items()}

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
