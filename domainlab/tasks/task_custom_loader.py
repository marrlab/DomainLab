"""
Use dictionaries to create train and test domain split
"""
from collections import Counter
import torch
from torch.utils.data.dataset import ConcatDataset

from domainlab.tasks.a_task import NodeTaskDGClassif
from domainlab.tasks.utils_task import mk_loader, mk_onehot, DsetDomainVecDecorator
from domainlab.tasks.utils_task_dset import DsetIndDecorator4XYD
from domainlab.tasks.b_task import NodeTaskDict


def mk_task_dset(dset_tr,
                 dset_val,
                 dset_te,
                 dict_domain2dset,
                 list_str_y,
                 isize,
                 taskna,  # name of the task
                 succ=None):
    class NodeTaskLoader(NodeTaskDict):
        """
        Use dictionaries to create train and test domain split
        """
        @property
        def list_str_y(self):
            return self._list_str_y

        @list_str_y.setter
        def list_str_y(self, list_str_y):
            self._list_str_y = list_str_y

        @property
        def isize(self):
            return self._im_size

        @isize.setter
        def isize(self, isize):
            self._im_size = isize

        def conf(self, args):
            self.list_str_y = list_str_y
            self.isize = isize

        def get_dset_by_domain(self, args, na_domain, split=None):
            return dict_domain2dset[na_domain]

        def init_business(self, args):
            """
            create a dictionary of datasets
            """
            self.conf(args)
            self.set_list_domains(list(dict_domain2dset.keys()))
            self._loader_tr = mk_loader(dset_tr, args.bs)
            self._loader_val = mk_loader(dset_val, args.bs)
            self._loader_te = mk_loader(dset_te, args.bs, drop_last=False)
            super().init_business(args)
    return NodeTaskLoader(succ=succ)
