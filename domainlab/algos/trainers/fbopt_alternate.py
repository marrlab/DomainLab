"""
update hyper-parameters during training
"""
import copy
import numpy as np
from domainlab.utils.logger import Logger


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
        self.rate_exp_shoulder = 0.0001
        self.epsilon_r = 20

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

    def search_mu(self, dict_theta=None, iter_start=None):
        """
        start from parameter dictionary dict_theta: {"layer":tensor},
        enlarge mu w.r.t. its current value
        to see if the criteria is met
        $$\\mu^{k+1}=mu^{k}exp(rate_mu*[R(\\theta^{k})-epsilon_R])$$
        """
        epo_reg_loss, _ = self.trainer.eval_r_loss()
        multiplier = np.exp(self.rate_exp_shoulder * (epo_reg_loss - self.epsilon_r))
        target = self.dict_multiply(self.mmu, multiplier)
        self.mmu = target
        return False

    def dict_is_zero(self, dict_mu):
        """
        check if hyper-parameter start from zero
        """
        for key in dict_mu.keys():
            if dict_mu[key] == 0.0:
                return True
        return False

    def dict_multiply(self, dict_base, multiplier):
        """
        multiply a float to each element of a dictionary
        """
        assert multiplier > 1
        return {key: val*multiplier for key, val in dict_base.items()}
