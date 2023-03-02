"""
Base Class for trainer
"""
import abc
from torch import optim


class AbstractTrainer(metaclass=abc.ABCMeta):
    """
    Algorithm director that controls the data flow
    """
    def __init__(self, model, task, observer, device, aconf, flag_accept=True):
        """
        model, task, observer, device, aconf
        """
        # @FIXME: aconf and args should be separated
        self.model = model
        self.task = task
        self.observer = observer
        self.device = device
        self.aconf = aconf
        #
        self.loader_tr = task.loader_tr
        self.loader_te = task.loader_te

        if flag_accept:
            self.observer.accept(self)

        self.model = self.model.to(device)
        #
        self.num_batches = len(self.loader_tr)
        self.flag_update_hyper_per_epoch = False
        self.flag_update_hyper_per_batch = False
        self.epo_loss_tr = None
        self.hyper_scheduler = None

    @abc.abstractmethod
    def tr_epoch(self, epoch):
        """
        :param epoch:
        """
        raise NotImplementedError

    def after_batch(self, epoch, ind_batch):
        """
        :param epoch:
        :param ind_batch:
        """
        return

    @abc.abstractmethod
    def before_tr(self):
        """
        before training, probe model performance
        """
        raise NotImplementedError

    def post_tr(self):
        """
        after training
        """
        self.observer.after_all()
