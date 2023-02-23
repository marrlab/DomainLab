"""
Abstract class for TaskClassif
"""
import os
import warnings
from abc import abstractmethod, abstractproperty

from domainlab.compos.pcr.p_chain_handler import AbstractChainNodeHandler
from domainlab.tasks.utils_task import img_loader2dir
from domainlab.tasks.task_utils import parse_domain_id


class NodeTaskDGClassif(AbstractChainNodeHandler):
    """
    Domain Generalization Classification Task
    """
    def __init__(self, succ=None):
        super().__init__(succ)
        self._loader_tr = None
        self._loader_te = None
        self._loader_val = None
        self._list_domains = None
        self._list_domain_tr = None
        self.dict_dset = {}
        self.dict_dset_te = {}
        self.dict_dset_val = {}
        self.dict_domain_class_count = {}
        self.dim_d_tr = None  # public
        self._list_str_y = None
        self._im_size = None
        self._dict_domains2imgroot = {}
        self._dict_domain_folder_name2class = {}  # {"domain1": {"class1":car, "class2":dog}}
        self._dict_domain_img_trans = {}
        self._dict_domain2filepath_list_im = {}  # {"photo": "xxx/yyy/file_of_path2imgs"}
        self.dict_att = {}
        self.img_trans_te = None

    @abstractmethod
    def init_business(self, args):
        """
        construct loader with resampling
        :param seed: random seed for resampling
        :param bs: batch size
        :param domain_na_tes: test domain names
        """

    @abstractmethod
    def get_list_domains(self):
        """
        1. get list of domain names
        2. better use method than property so new domains can be added
        """
        raise NotImplementedError

    @abstractproperty
    def list_str_y(self):
        raise NotImplementedError

    @abstractproperty
    def isize(self):
        """image channel, height, width"""
        raise NotImplementedError

    ###########################################################################
    @property
    def dim_y(self):
        """classification dimension"""
        return len(self.list_str_y)

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
        # @FIXME: hardcoded position
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
            warnings.warn(
                "The sets of training and test domains overlap -- \
                be aware of data leakage or training to the test!",
                RuntimeWarning
            )

        self.dim_d_tr = len(list_domain_tr)
        self._list_domain_tr = list_domain_tr
        return list_domain_tr, list_domain_te

    def sample_sav(self, root, batches=5, subfolder_na="task_sample"):
        """
        sample data from task and save to disk
        """
        folder_na = os.path.join(root, self.task_name, subfolder_na)

        img_loader2dir(self.loader_te,
                       list_domain_na=self.get_list_domains(),
                       list_class_na=self.list_str_y,
                       folder=folder_na,
                       batches=batches,
                       test=True)

        img_loader2dir(self.loader_tr,
                       list_domain_na=self.get_list_domains(),
                       list_class_na=self.list_str_y,
                       folder=folder_na,
                       batches=batches)
