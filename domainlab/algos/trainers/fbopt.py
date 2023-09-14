"""
update hyper-parameters during training
"""
import copy
import numpy as np
from domainlab.utils.logger import Logger


class HyperSchedulerFeedback():
    """
    design $\\mu$$ sequence based on state of penalized loss
    """
    def __init__(self, trainer, **kwargs):
        """
        kwargs is a dictionary with key the hyper-parameter name and its value
        """
        self.trainer = trainer
        self.mmu = kwargs
        self.mmu = {key: 0.0 for key, val in self.mmu.items()}
        self.ploss_old_theta_old_mu = None
        self.ploss_old_theta_new_mu = None
        self.ploss_new_theta_old_mu = None
        self.ploss_new_theta_new_mu = None
        self.delta_mu = trainer.aconf.delta_mu
        self.init_mu = trainer.aconf.init_mu4beta
        # for exponential increase of mu, mu can not be starting from zero
        self.beta_mu = trainer.aconf.beta_mu
        self.dict_theta_bar = None
        self.dict_theta = None
        # theta_ref should be equal to either theta or theta bar as reference
        # since theta_ref will be used to judge if criteria is met
        self.dict_theta_ref = None
        self.budget_mu_per_step = trainer.aconf.budget_mu_per_step
        self.budget_theta_update_per_mu = trainer.aconf.budget_theta_update_per_mu
        self.count_found_operator = 0
        self.count_search_mu = 0

    def search_mu(self, dict_theta, iter_start):
        """
        start from parameter dictionary dict_theta: {"layer":tensor},
        enlarge mu w.r.t. its current value
        to see if the criteria is met
        """
        self.count_search_mu += 1
        self.dict_theta = copy.deepcopy(dict_theta)
        self.dict_theta_ref = self.dict_theta
        # I am not sure if necessary here to deepcopy, but safer
        logger = Logger.get_logger(logger_name='main_out_logger', loglevel="INFO")
        mmu = None
        # self.mmu is not updated until a reg-descent operator is found
        self.ploss_old_theta_old_mu = self.trainer.eval_p_loss(self.mmu, self.dict_theta_ref)
        for miter in range(iter_start, self.budget_mu_per_step):
            mmu = self.dict_mu_iter(miter)
            print(f"trying mu={mmu} at mu iteration {miter} of {self.budget_mu_per_step}")
            if self.search_theta(mmu):
                self.count_found_operator += 1
                logger.info(f"!!!found reg-pareto operator with mu={mmu}")
                logger.info(f"success rate: {self.count_found_operator}/{self.count_search_mu}")
                return True
        logger.warn(f"!!!!!!failed to find mu within budget, mu={mmu}")
        logger.info(f"success rate: {self.count_found_operator}/{self.count_search_mu}")
        return False

    def search_theta(self, mmu_new):
        """
        conditioned on fixed $$\\mu$$, the operator should search theta based on
        the current value of $theta$

        the execution will set the value for mu and theta as well
        """
        self.ploss_old_theta_new_mu = self.trainer.eval_p_loss(mmu_new, self.dict_theta_ref)
        theta4mu_new = copy.deepcopy(self.dict_theta)
        for i in range(self.budget_theta_update_per_mu):
            print(f"search theta at iteration {i} with mu={mmu_new}")
            theta4mu_new = self.trainer.opt_theta(mmu_new, theta4mu_new)
            self.ploss_new_theta_new_mu = self.trainer.eval_p_loss(mmu_new, theta4mu_new)
            self.ploss_new_theta_old_mu = self.trainer.eval_p_loss(self.mmu, theta4mu_new)
            if self.is_criteria_met():
                self.mmu = mmu_new
                self.dict_theta = theta4mu_new
                return True
        return False

    def is_criteria_met(self):
        """
        if the reg-descent criteria is met
        """
        flag_improve = self.ploss_new_theta_new_mu < self.ploss_old_theta_new_mu
        flag_deteriorate = self.ploss_new_theta_old_mu > self.ploss_old_theta_old_mu
        return flag_improve & flag_deteriorate

    def dict_mu_iter(self, miter):
        """
        update the dictionary of mu w.r.t. its current value, and its iteration, and its iteration
        """
        if miter == 0:
            return self.mmu
        if self.delta_mu is not None:
            mmu = self.dict_addition(self.mmu, miter * self.delta_mu)
        elif self.beta_mu is not None:
            if self.dict_is_zero(self.mmu):
                base = self.dict_addition(self.mmu, self.init_mu)
            else:
                base = self.mmu
            multiplier = np.power(self.beta_mu, miter)
            mmu = self.dict_multiply(base, multiplier)
        else:
            raise RuntimeError("delta_mu and beta_mu can not be simultaneously None!")
        return mmu

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

    def dict_addition(self, dict_base, delta):
        """
        increase the value of a dictionary by delta
        """
        return {key: val + delta for key, val in dict_base.items()}
