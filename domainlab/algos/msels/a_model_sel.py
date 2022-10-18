import abc

class AMSel(metaclass=abc.ABCMeta):
    def __init__(self):
        self.trainer = None
        self.tr_obs = None

    def accept(self, trainer, tr_obs):
        """
        Visitor pattern to trainer
        """
        self.trainer = trainer
        self.tr_obs = tr_obs

    @abc.abstractmethod
    def update(self):
        """
        observer + visitor pattern to trainer
        """
        raise NotImplementedError

    def if_stop(self):
        """
        check if trainer should stop
        """
        raise NotImplementedError
