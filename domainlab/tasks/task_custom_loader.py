"""
Use dictionaries to create train and test domain split
"""
from collections import Counter
import torch
from torch.utils.data.dataset import ConcatDataset

from domainlab.tasks.a_task import NodeTaskDGClassif
from domainlab.tasks.utils_task import mk_loader, mk_onehot, DsetDomainVecDecorator
from domainlab.tasks.utils_task_dset import DsetIndDecorator4XYD
from domainlab.tasks.task_loader import NodeTaskDict


def dset_decoration_args_algo(args, ddset):
    if "match" in args.aname:  # FIXME: are there ways not to use this if statement?
            ddset = DsetIndDecorator4XYD(ddset)
    return ddset


class NodeTaskLoader(NodeTaskDict):
    """
    Use dictionaries to create train and test domain split
    """
    def init_business(self, args):
        """
        create a dictionary of datasets
        """
        self._loader_tr = mk_loader(ddset_mix, args.bs)
        self._loader_val = mk_loader(ddset_mix_val, args.bs)
        self._loader_te = mk_loader(dset_te, args.bs, drop_last=False)
        self.count_domain_class()

    def count_domain_class(self):
        """
        iterate all domains and count the class label distribution for each
        return a double dictionary {"domain1": {"class1":3, "class2": 4,...}, ....}
        """
        for key, dset in self.dict_dset.items():
            dict_class_count = self._count_class_one_hot(dset)
            self.dict_domain_class_count[key] = dict_class_count
        for key, dset in self.dict_dset_te.items():
            dict_class_count = self._count_class_one_hot(dset)
            self.dict_domain_class_count[key] = dict_class_count

    def _count_class_one_hot(self, dset):
        labels_count = torch.zeros(self.dim_y, dtype=torch.long)
        for _, target, *_ in dset:
            labels_count += target.long()

        list_count = list(labels_count.cpu().numpy())
        dict_class_count = dict()
        for name, count in zip(self.list_str_y, list_count):
            dict_class_count[name] = count
        return dict_class_count
