"""
Base Class for trainer
"""
import abc
from torch import optim
from domainlab.compos.pcr.p_chain_handler import AbstractChainNodeHandler

def mk_opt(model, aconf):
    """
    create optimizer
    """
    optimizer = optim.Adam(model.parameters(), lr=aconf.lr)
    return optimizer


class AbstractTrainer(AbstractChainNodeHandler, metaclass=abc.ABCMeta):
    """
    Algorithm director that controls the data flow
    """
    @property
    def p_na_prefix(self):
        """
        common prefix for Trainers
        """
        return "Trainer"

    def __init__(self, successor_node=None):
        """__init__.
        :param successor_node:
        """
        super().__init__(successor_node)
        self.model = None
        self.task = None
        self.observer = None
        self.device = None
        self.aconf = None
        #
        self.loader_tr = None
        self.loader_te = None
        self.num_batches = None
        self.flag_update_hyper_per_epoch = None
        self.flag_update_hyper_per_batch = None
        self.epo_loss_tr = None
        self.hyper_scheduler = None
        self.optimizer = None
        self.exp = None
        self.args = None
        self.ctr_model = None
        self.erm = None
        # mldg
        self.inner_trainer = None
        self.loader_tr_source_target = None


    def init_business(self, model, task, observer, device, aconf, flag_accept=True):
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
        self.optimizer = mk_opt(self.model, self.aconf)

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

    @property
    def name(self):
        """
        get the name of the algorithm
        """
        na_prefix = self.p_na_prefix
        len_prefix = len(na_prefix)
        na_class = type(self).__name__
        if na_class[:len_prefix] != na_prefix:
            raise RuntimeError(
                "Trainer builder node class must start with ", na_prefix,
                "the current class is named: ", na_class)
        return type(self).__name__[len_prefix:].lower()

    def is_myjob(self, request):
        """
        :param request: string
        """
        return request == self.name
