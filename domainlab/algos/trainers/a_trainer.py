"""
Base Class for trainer
"""
import abc
from domainlab.utils.perf import PerfClassif


class AbstractTrainer(metaclass=abc.ABCMeta):
    """
    Algorithm director that controls the data flow
    """
    def __init__(self, model, task, observer, device, aconf):
        """
        model, task, observer, device, aconf
        """
        # FIXME: aconf and args should be separated
        self.model = model
        self.task = task
        self.observer = observer
        self.device = device
        self.aconf = aconf
        #
        self.loader_tr = task.loader_tr
        self.loader_te = task.loader_te
        self.observer.accept(self)
        self.model = self.model.to(device)

    @abc.abstractmethod
    def tr_epoch(self, epoch):
        """
        :param epoch:
        """
        raise NotImplementedError

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


class TrainerClassif(AbstractTrainer):
    """
    Base class for trainer of classification task
    """
    def tr_epoch(self, epoch):
        """
        :param epoch:
        """
        raise NotImplementedError

    def before_tr(self):
        """
        check the performance of randomly initialized weight
        """
        acc = PerfClassif.cal_acc(self.model, self.loader_te, self.device)
        print("before training, model accuracy:", acc)
