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
    optimizer = optim.Adam(model.as_model().parameters(), lr=aconf.lr)
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

    def extend(self, trainer):
        """
        extend current trainer with another trainer
        """
        self._decoratee = trainer

    def __init__(self, successor_node=None, extend=None):
        """__init__.
        :param successor_node:
        """
        super().__init__(successor_node)
        self._model = None
        self._decoratee = extend
        self.task = None
        self.observer = None
        self.device = None
        self.aconf = None
        self.gamma_reg = None
        #
        self.loader_tr = None
        self.loader_tr_no_drop = None
        self.loader_te = None
        self.num_batches = None
        self.flag_update_hyper_per_epoch = None
        self.flag_update_hyper_per_batch = None
        self.epo_loss_tr = None
        self.epo_reg_loss_tr = None
        self.epo_task_loss_tr = None
        self.counter_batch = None
        self.hyper_scheduler = None
        self.optimizer = None
        self.exp = None
        self.args = None
        self.ctr_model = None
        self.erm = None
        # mldg
        self.inner_trainer = None
        self.loader_tr_source_target = None
        self.flag_initialized = False
        # fbopt
        self.mu_iter_start = 0
        self.flag_setpoint_updated = False


    @property
    def model(self):
        """
        property model, which can be another trainer or model
        """
        return self.get_model()

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def str_metric4msel(self):
        """
        metric for model selection
        """
        return self.model.metric4msel

    def init_business(self, model, task, observer, device, aconf, flag_accept=True):
        """
        model, task, observer, device, aconf
        """
        if self._decoratee is not None:
            self._decoratee.init_business(model, task, observer, device, aconf, flag_accept)
            self._model = self._decoratee
        else:
            self._model = model
        self.task = task
        self.observer = observer
        self.device = device
        self.aconf = aconf
        self.gamma_reg = self.aconf.gamma_reg
        #
        self.loader_tr = task.loader_tr
        self.loader_tr_no_drop = task._loader_tr_no_drop
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
        self.reset()
        self.flag_initialized = True

    def reset(self):
        """
        make a new optimizer to clear internal state
        """
        self.optimizer = mk_opt(self.model, self.aconf)

    @abc.abstractmethod
    def tr_epoch(self, epoch):
        """
        :param epoch:
        """
        raise NotImplementedError

    def before_batch(self, epoch, ind_batch):
        """
        :param epoch:
        :param ind_batch:
        """
        return

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

    def get_model(self):
        """
        recursively get the "real" model from trainer
        """
        if "trainer" not in str(type(self._model)).lower():
            return self._model
        return self._model.get_model()
      
    def as_model(self):
        """
        used for decorator pattern

        It is not necessary to write any function that just copies the pattern
        self.get_model().do_something()
        """
        return self.get_model()

    def cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        """
        decorate trainer regularization loss
        combine losses of current trainer with self._model.cal_reg_loss, which
        can be either a trainer or a model
        """
        list_reg_model, list_mu_model = self._model.cal_reg_loss(
            tensor_x, tensor_y, tensor_d, others)
        list_reg, list_mu = self._cal_reg_loss(tensor_x, tensor_y, tensor_d, others)
        return list_reg_model + list_reg, list_mu_model + list_mu

    def _cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        """
        interface for each trainer to implement
        """
        return [], []
