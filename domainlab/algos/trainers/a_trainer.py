"""
Base Class for trainer
"""
import abc

import torch
from torch import optim
from torch.optim import lr_scheduler 

from domainlab.compos.pcr.p_chain_handler import AbstractChainNodeHandler


def mk_opt(model, aconf):
    """
    create optimizer
    """
    if model._decoratee is None:
        class_opt = getattr(optim, aconf.opt)
        optimizer = class_opt(model.parameters(), lr=aconf.lr)
    else:
        var1 = model.parameters()
        var2 = model._decoratee.parameters()
        set_param = set(list(var1) + list(var2))
        list_par = list(set_param)
        # optimizer = optim.Adam([var1, var2], lr=aconf.lr)
        # optimizer = optim.Adam([
        #    {'params': model.parameters()},
        #    {'params': model._decoratee.parameters()}
        # ], lr=aconf.lr)
        optimizer = optim.Adam(list_par, lr=aconf.lr)
    if aconf.lr_scheduler is not None:
        class_lr_scheduler = getattr(lr_scheduler, aconf.lr_scheduler)
        scheduler = class_lr_scheduler(optimizer, T_max=aconf.epos)
    else:
        scheduler = None
    return optimizer, scheduler


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
        #
        self.dict_loader_tr = None
        self.loader_tr = None
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
        # matchdg
        self.lambda_ctr = None
        self.flag_stop = None
        self.flag_erm = None
        self.tensor_ref_domain2each_domain_x = None
        self.tensor_ref_domain2each_domain_y = None
        self.base_domain_size = None
        self.tuple_tensor_ref_domain2each_y = None
        self.tuple_tensor_refdomain2each = None
        # mldg
        self.inner_trainer = None
        self.loader_tr_source_target = None
        self.flag_initialized = False
        # moving average
        self.ma_weight_previous_model_params = None
        self._dict_previous_para_persist = {}
        self._ma_iter = 0
        #
        self.list_reg_over_task_ratio = None
        # MIRO
        self.input_tensor_shape = None
        # LR-scheduler
        self.lr_scheduler = None

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

    @property
    def list_tr_domain_size(self):
        """
        get a list of training domain size
        """
        train_domains = self.task.list_domain_tr
        return [len(self.task.dict_dset_tr[key]) for key in train_domains]

    @property
    def decoratee(self):
        if self._decoratee is None:
            return self.model
        return self._decoratee

    def init_business(self, model, task, observer, device, aconf, flag_accept=True):
        """
        model, task, observer, device, aconf
        """
        # Note self.decoratee can be both model and trainer,
        # but self._decoratee can only be trainer!
        if self._decoratee is not None:
            self._decoratee.init_business(
                model, task, observer, device, aconf, flag_accept
            )
            self.model = self._decoratee
        else:
            self.model = model
        self.task = task
        self.task.init_business(trainer=self, args=aconf)
        self.model.list_d_tr = self.task.list_domain_tr
        self.observer = observer
        self.device = device
        self.aconf = aconf
        #
        self.dict_loader_tr = task.dict_loader_tr
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
        self.reset()
        self.flag_initialized = True

    def reset(self):
        """
        make a new optimizer to clear internal state
        """
        self.optimizer, self.lr_scheduler = mk_opt(self.model, self.aconf)

    @abc.abstractmethod
    def tr_epoch(self, epoch):
        """
        :param epoch:
        """

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

    def before_tr(self):
        """
        before training, probe model performance
        """
        self.cal_reg_loss_over_task_loss_ratio()

    def cal_reg_loss_over_task_loss_ratio(self):
        """
        estimate the scale of each loss term, match each loss term to the major
        loss via a ratio, this ratio will be multiplied with multiplier
        """
        list_accum_reg_loss = []
        loss_task_agg = 0
        for ind_batch, (tensor_x, tensor_y, tensor_d, *others) in enumerate(
            self.loader_tr
        ):
            self.input_tensor_shape = tensor_x.shape
            if ind_batch >= self.aconf.nb4reg_over_task_ratio:
                return
            tensor_x, tensor_y, tensor_d = (
                tensor_x.to(self.device),
                tensor_y.to(self.device),
                tensor_d.to(self.device),
            )
            list_reg_loss_tensor, _ = \
                self.cal_reg_loss(tensor_x, tensor_y, tensor_d, others)
            list_reg_loss_tensor = [torch.sum(tensor).detach().item()
                                    for tensor in list_reg_loss_tensor]
            if ind_batch == 0:
                list_accum_reg_loss = list_reg_loss_tensor
            else:
                list_accum_reg_loss = [reg_loss_accum_tensor + reg_loss_tensor
                                       for reg_loss_accum_tensor,
                                       reg_loss_tensor in
                                       zip(list_accum_reg_loss,
                                           list_reg_loss_tensor)]
            tensor_loss_task = self.model.cal_task_loss(tensor_x, tensor_y)
            tensor_loss_task = torch.sum(tensor_loss_task).detach().item()
            loss_task_agg += tensor_loss_task
        self.list_reg_over_task_ratio = [reg_loss / loss_task_agg
                                         for reg_loss in list_accum_reg_loss]

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
                "Trainer builder node class must start with ",
                na_prefix,
                "the current class is named: ",
                na_class,
            )
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

    def cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        """
        decorate trainer regularization loss
        combine losses of current trainer with self._model.cal_reg_loss, which
        can be either a trainer or a model
        """
        list_reg_loss_model_tensor, list_mu_model = \
            self.decoratee.cal_reg_loss(tensor_x, tensor_y, tensor_d, others)
        assert len(list_reg_loss_model_tensor) == len(list_mu_model)

        list_reg_loss_trainer_tensor, list_mu_trainer = self._cal_reg_loss(
            tensor_x, tensor_y, tensor_d, others
        )
        assert len(list_reg_loss_trainer_tensor) == len(list_mu_trainer)
        # extend the length of list: extend number of regularization loss
        # tensor: the element of list is tensor
        list_loss_tensor = list_reg_loss_model_tensor + \
            list_reg_loss_trainer_tensor
        list_mu = list_mu_model + list_mu_trainer
        return list_loss_tensor, list_mu

    def _cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        """
        interface for each trainer to implement
        """
        return [], []

    def dset_decoration_args_algo(self, args, ddset):
        """
        decorate dataset to get extra entries in load item, for instance,
        jigen need permutation index
        this parent class function delegate decoration to its decoratee
        """
        if self._decoratee is not None:
            return self._decoratee.dset_decoration_args_algo(args, ddset)
        return ddset

    def print_parameters(self):
        """
        Function to print all parameters of the object.
        Can be used to print the parameters of any child class
        """
        params = vars(self)
        print(f"Parameters of {type(self).__name__}: {params}")
