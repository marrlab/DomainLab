"""
Use dictionaries to create train and test domain split
"""
from collections import Counter

import torch
from torch.utils.data.dataset import ConcatDataset

from domainlab.tasks.a_task import NodeTaskDGClassif
from domainlab.tasks.utils_task import (DsetDomainVecDecorator, mk_loader,
                                        mk_onehot)
from domainlab.tasks.utils_task_dset import DsetIndDecorator4XYD
from domainlab.dsets.utils_wrapdset_patches import WrapDsetPatches


def dset_decoration_args_algo(args, ddset):
    if "match" in args.aname:  # @FIXME: are there ways not to use this if statement?
        ddset = DsetIndDecorator4XYD(ddset)
    if "jigen" in args.aname:
        # FIXME: do this during before_tr
        ddset = WrapDsetPatches(ddset,
                                num_perms2classify=args.nperm,
                                prob_no_perm=args.pperm,
                                grid_len=args.grid_len)
    return ddset


class NodeTaskDict(NodeTaskDGClassif):
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
    def isize(self, im_size):
        self._im_size = im_size

    def get_list_domains(self):
        return self._list_domains

    def set_list_domains(self, list_domains):
        self._list_domains = list_domains

    def get_dset_by_domain(self, args, na_domain):
        raise NotImplementedError

    def init_business(self, args):
        """
        create a dictionary of datasets
        """
        list_domain_tr, list_domain_te = self.get_list_domains_tr_te(args.tr_d, args.te_d)
        self.dict_dset = dict()
        self.dict_dset_val = dict()
        dim_d = len(list_domain_tr)
        for (ind_domain_dummy, na_domain) in enumerate(list_domain_tr):
            dset_tr, dset_val = self.get_dset_by_domain(args, na_domain)
            # @FIXME: currently, different task has different default values for
            # split, for TaskFolder split default to False, for mnist, split
            # default to True
            vec_domain = mk_onehot(dim_d, ind_domain_dummy)
            ddset_tr = DsetDomainVecDecorator(dset_tr, vec_domain, na_domain)
            ddset_val = DsetDomainVecDecorator(dset_val, vec_domain, na_domain)
            ddset_tr = dset_decoration_args_algo(args, ddset_tr)
            ddset_val = dset_decoration_args_algo(args, ddset_val)
            self.dict_dset.update({na_domain: ddset_tr})
            self.dict_dset_val.update({na_domain: ddset_val})
        ddset_mix = ConcatDataset(tuple(self.dict_dset.values()))
        self._loader_tr = mk_loader(ddset_mix, args.bs)

        ddset_mix_val = ConcatDataset(tuple(self.dict_dset_val.values()))
        self._loader_val = mk_loader(ddset_mix_val, args.bs)

        self.dict_dset_te = dict()
        # No need to have domain Label for test
        for na_domain in list_domain_te:
            dset_te, *_ = self.get_dset_by_domain(args, na_domain, split=False)
            # @FIXME: since get_dset_by_domain always return two datasets,
            # train and validation, this is not needed in test domain
            self.dict_dset_te.update({na_domain: dset_te})
        dset_te = ConcatDataset(tuple(self.dict_dset_te.values()))
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

    def _count_class(self, dset):   # @FIXME: remove this
        labels = dset.targets
        class_dict = dict(Counter(labels))
        return class_dict
