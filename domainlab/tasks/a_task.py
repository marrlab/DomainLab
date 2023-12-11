"""
Base class for Task
"""
from abc import abstractmethod
import warnings

from domainlab.compos.pcr.p_chain_handler import AbstractChainNodeHandler
from domainlab.tasks.task_utils import parse_domain_id
from domainlab.utils.logger import Logger


class NodeTaskDG(AbstractChainNodeHandler):
    """
    Domain Generalization Classification Task
    """
    def __init__(self, succ=None):
        super().__init__(succ)
        self._loader_tr = None
        self._loader_te = None
        self._loader_val = None
        self._list_domains = None
        self._list_domain_tr = None  # versatile
        self._name = None
        self._args = None
        self.dict_dset_all = {}  # persist
        self.dict_dset_tr = {}  # versatile variable: which domains to use as training
        self.dict_dset_te = {}  # versatile
        self.dict_dset_val = {}  # versatile
        self.dict_domain_class_count = {}
        self.dim_d_tr = None  # public, only used for diva
        self._im_size = None
        self._dict_domains2imgroot = {}
        self._dict_domain_folder_name2class = {}  # {"domain1": {"class1":car, "class2":dog}}
        self._dict_domain_img_trans = {}
        self.dict_att = {}
        self.img_trans_te = None
        self.dict_domain2imgroot = {}
        self._dict_domain2filepath_list_im_tr = {}  # {"photo": "xxx/yyy/file_of_path2imgs"}
        self._dict_domain2filepath_list_im_val = {}
        self._dict_domain2filepath_list_im_te = {}
        self.dict_class_label_ind2name = None
        self.conf_without_args()  # configuration without init_business

    def conf_without_args(self):
        """
        configuration without init_business
        """

    @abstractmethod
    def init_business(self, args, trainer=None):
        """
        construct task data loader
        """

    def get_list_domains(self):
        """
        1. get list of domain names
        2. better use method than property so new domains can be added
        """
        return self._list_domains

    def set_list_domains(self, list_domains):
        """
        setter for self._list_domains
        """
        self._list_domains = list_domains

    @property
    def isize(self):
        """
        getter for input size: isize
        """
        return self._im_size

    @isize.setter
    def isize(self, im_size):
        """
        setter for input size: isize
        """
        self._im_size = im_size

    @property
    def list_domain_tr(self):
        """
        property getter of list of domains for this task
        """
        if self._list_domain_tr is None:
            raise RuntimeError("task not intialized!")
        return self._list_domain_tr

    @property
    def loader_tr(self):
        """loader of mixed train domains"""
        return self._loader_tr

    @property
    def loader_val(self):
        """loader of validation dataset on the training domains"""
        return self._loader_val

    @property
    def loader_te(self):
        """loader of mixed test domains"""
        return self._loader_te

    @property
    def task_name(self):
        """
        The basic name of the task, without configurations
        """
        # @FIXME: hardcoded position:NodeTaskXXX
        return type(self).__name__[8:].lower()

    def get_na(self, na_tr, na_te):
        """
        task name appended with configurations
        :param na_tr: training domain names
        :param na_te: test domain names
        """
        _, list_te = self.get_list_domains_tr_te(na_tr, na_te)
        str_te = "_".join(list_te)
        # train domain names are too long
        return "_".join([self.task_name, "te", str_te])

    def is_myjob(self, request):
        """
        :param request: string
        """
        return request == self.task_name

    def get_list_domains_tr_te(self, tr_id, te_id):
        """
        For static DG task, get train and test domains list.

        :param tr_id: training domain ids;
            int or str, or a list of int or str, or None;
            if None, then assumed to be the complement of te_id.
        :param te_id: test domain ids;
            int or str, or a list of int or str; required.
        :return: list of training domain names, list of test domain names.
        """
        list_domains = self.get_list_domains()

        list_domain_te = parse_domain_id(te_id, list_domains)
        assert set(list_domain_te).issubset(set(list_domains))

        if tr_id is None:
            list_domain_tr = [did for did in list_domains if
                              did not in list_domain_te]
        else:
            list_domain_tr = parse_domain_id(tr_id, list_domains)
        if not set(list_domain_tr).issubset(set(list_domains)):
            raise RuntimeError(
                f"training domain {list_domain_tr} is not \
                subset of available domains {list_domains}")

        if set(list_domain_tr) & set(list_domain_te):
            logger = Logger.get_logger()
            logger.warn(
                "The sets of training and test domains overlap -- "
                "be aware of data leakage or training to the test!"
            )
            warnings.warn(
                "The sets of training and test domains overlap -- "
                "be aware of data leakage or training to the test!",
                RuntimeWarning
            )

        self.dim_d_tr = len(list_domain_tr)
        self._list_domain_tr = list_domain_tr
        return list_domain_tr, list_domain_te

    def __str__(self):
        """
        print the attribute of the task
        """
        strout = "list of domains: \n"
        strout += str(self.get_list_domains())
        strout += (f"\n input tensor size: {self.isize}")
        return strout
