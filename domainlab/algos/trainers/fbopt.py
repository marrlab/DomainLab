"""
update hyper-parameters during training
"""
import copy
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
        self.delta_mu = 0.01   # FIXME
        self.dict_theta = None
        self.budget_mu_per_step = 5  # FIXME
        self.budget_theta_update_per_mu = 5  # np.infty

    def search_mu(self, dict_theta, iter_start=0):
        """
        start from parameter dict_theta,
        enlarge mmu to see if the criteria is met
        """
        self.dict_theta = dict_theta
        mmu = None
        for miter in range(iter_start, self.budget_mu_per_step):
            mmu = self.dict_addition(self.mmu, miter * self.delta_mu)
            print(f"trying mu={mmu} at mu iteration {miter}")
            if self.search_theta(mmu):
                print(f"!!!found reg-pareto operator with mu={mmu}")
                self.mmu = mmu
                return True
        logger = Logger.get_logger(logger_name='main_out_logger', loglevel="INFO")
        logger.warn(f"!!!!!!failed to find mu within budget, mu={mmu}")
        return False

    def dict_addition(self, dict_base, delta):
        """
        increase the value of a dictionary by delta
        """
        return {key: val + delta for key, val in dict_base.items()}

    def search_theta(self, mmu_new):
        """
        conditioned on fixed $$\\mu$$, the operator should search theta based on
        the current value of $theta$

        the execution will set the value for mu and theta as well
        """
        flag_success = False
        self.ploss_old_theta_new_mu = self.trainer.eval_loss(mmu_new, self.dict_theta)
        self.ploss_old_theta_old_mu = self.trainer.eval_loss(self.mmu, self.dict_theta)
        theta4mu_new = copy.deepcopy(self.dict_theta)
        for i in range(self.budget_theta_update_per_mu):
            print(f"update theta at iteration {i} with mu={mmu_new}")
            theta4mu_new = self.trainer.opt_theta(mmu_new, theta4mu_new)
            self.ploss_new_theta_new_mu = self.trainer.eval_loss(mmu_new, theta4mu_new)
            self.ploss_new_theta_old_mu = self.trainer.eval_loss(self.mmu, theta4mu_new)
            if self.is_criteria_met():
                self.mmu = mmu_new
                flag_success = True
                # FIXME: update theta only if current mu is good enough?
                self.dict_theta = theta4mu_new
                return flag_success
        return flag_success

    def inner_product(self, mmu, v_reg_loss):
        """
        - the first dimension of the tensor v_reg_loss is mini-batch
        the second dimension is the number of regularizers
        - the vector mmu has dimension the number of regularizers
        """
        return mmu * v_reg_loss  #

    def is_criteria_met(self):
        """
        if the reg-descent criteria is met
        """
        flag_improve = self.ploss_new_theta_new_mu < self.ploss_old_theta_new_mu
        flag_deteriorate = self.ploss_new_theta_old_mu > self.ploss_old_theta_old_mu
        return flag_improve & flag_deteriorate

    def __call__(self, epoch):
        """
        """