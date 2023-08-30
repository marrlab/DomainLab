class HyperSchedulerFeedback():
    """
    HyperSchedulerWarmup
    """
    def __init__(self, **kwargs):
        """
        kwargs is a dictionary with key the hyper-parameter name and its value
        """
        self.dict_par_init = kwargs
        self.state = State()

    def feedback(self, epoch):
        """
        start from a small value of par to ramp up the steady state value using
        # total_steps
        :param epoch:
        """
        par_val = 0
        return par_val

    def __call__(self, epoch):
        dict_rst = {}
        for key, _ in self.dict_par_init.items():
            dict_rst[key] = self.feedback(epoch)
        return dict_rst




Class State():
    """
    state design pattern to memorize the historical loss
    """
    def observe(self):
