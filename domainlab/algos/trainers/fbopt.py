"""
update hyper-parameters during training
"""


class HyperSchedulerFeedback():
    """
    design $\\mu$$ sequence based on state of penalized loss
    """
    def __init__(self, **kwargs):
        """
        kwargs is a dictionary with key the hyper-parameter name and its value
        """
        self.dict_par_init = kwargs
        self.ploss_old_theta_old_mu = None
        self.ploss_old_theta_new_mu = None
        self.ploss_new_theta_old_mu = None
        self.ploss_new_theta_new_mu = None
        self.delta_mu = 0.01   # FIXME
        self.theta = None
        self.mmu = None
        self.opt_theta = None  # theta operator
        self.budget_mu_per_step = 5  # FIXME
        self.budget_theta_per_step = 5

    def search_mu(self):
        """
        enlarge mmu to see if the criteria is met
        """
        flag_success = False
        for miter in range(self.budget_mu_per_step):
            mmu = self.mmu + miter * self.delta_mu
            if self.search_theta(mmu):
                flag_success = True
        if not flag_success:
            raise RuntimeError("failed to find mu within budget")

    def search_theta(self, mmu_new):
        """
        conditioned on fixed $$\\mu$$, the operator should search theta based on
        the current value of $theta$
        """
        flag_success = False
        self.ploss_old_theta_new_mu = self.eval_loss(mmu_new, self.theta)
        self.ploss_old_theta_old_mu = self.eval_loss(self.mmu, self.theta)
        for _ in range(self.budget_theta_per_step):
            theta4mu_new = self.opt_theta(mmu_new)
            self.ploss_new_theta_new_mu = self.eval_loss(mmu_new, theta4mu_new)
            self.ploss_new_theta_old_mu = self.eval_loss(self.mmu, theta4mu_new)
            if self.is_criteria_met():
                self.mmu = mmu_new
                self.theta = theta4mu_new
                flag_success = True
        return flag_success

    def eval_loss(self, mmu, theta):
        """
        evaluate the penalty function value
        """
        v_reg_loss = self.trainer.model.cal_reg_loss(theta)
        loss_reg = self.inner_product(v_reg_loss, mmu)
        ploss = loss_reg + self.trainer.model.cal_task_loss(theta)
        return ploss

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
