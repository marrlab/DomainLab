"""
Abstract class for TaskClassif
"""
import os
from abc import abstractmethod, abstractproperty
from libdg.compos.pcr.p_chain_handler import AbstractChainNodeHandler
from libdg.tasks.utils_task import img_loader2dir


class NodeTaskDGClassif(AbstractChainNodeHandler):
    """
    Domain Generalization Classification Task
    """
    def __init__(self, succ=None):
        super().__init__(succ)
        self._loader_tr = None
        self._loader_te = None
        self._list_domains = None
        self._list_domain_tr = None
        self.dict_dset = dict()
        self.dict_dset_te = dict()
        self.dict_domain_class_count = {}
        self.dim_d_tr = None  # public
        self._list_str_y = None
        self._im_size = None
        self._dict_domains2imgroot = {}
        self._dict_domain_folder_name2class = {}  # {"domain1": {"class1":car, "class2":dog}}
        self._dict_domain_img_trans = {}
        self._dict_domain2filepath_list_im = {}  # {"photo": "xxx/yyy/file_of_path2imgs"}
        self.dict_att = {}

    @abstractmethod
    def init_business(self, args):
        """
        construct loader with resampling
        :param seed: random seed for resampling
        :param bs: batch size
        :param domain_na_tes: test domain names
        """
        raise NotImplementedError

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
        if self._list_domain_tr is None:
            raise RuntimeError("task not intialized!")
        return self._list_domain_tr

    @property
    def loader_tr(self):
        """loader of mixed train domains"""
        return self._loader_tr

    @property
    def loader_te(self):
        """loader of mixed test domains"""
        return self._loader_te


    @property
    def task_name(self):
        """
        The basic name of the task, without configurations
        """
        return type(self).__name__[8:].lower()

    def get_na(self, na_te):
        """
        task name appended with configurations
        """
        list_tr, list_te = self.get_list_domains_tr_te(na_te)
        str_tr = "_".join(list_tr)
        str_te = "_".join(list_te)
        return "_".join([self.task_name, "te", str_te])  # train domain names are too long

    def is_myjob(self, request):
        """
        :param request: string
        """
        return request == self.task_name

    def get_list_domains_tr_te(self, te_id):
        """
        For static DG task, get train and test domains list
        """
        list_domains = self.get_list_domains()
        if not isinstance(te_id, list):
            te_id = [te_id]

        list_domain_te = []
        for ele in te_id:
            if isinstance(ele, int):
                list_domain_te.append(list_domains[ele])
            elif isinstance(ele, str):
                if ele.isdigit():
                    list_domain_te.append(list_domains[int(ele)])
                else:
                    list_domain_te.append(ele)
            else:
                raise RuntimeError("test domain should be either int or str")
        assert set(list_domain_te).issubset(set(list_domains))
        list_domain_tr = [did for did in list_domains if did not in list_domain_te]
        assert set(list_domain_tr).issubset(set(list_domains))
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
                       batches=batches)

        img_loader2dir(self.loader_tr,
                       list_domain_na=self.get_list_domains(),
                       list_class_na=self.list_str_y,
                       folder=folder_na,
                       batches=batches)
