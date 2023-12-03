"""
Use dictionaries to create train and test domain split
"""
from torch.utils.data.dataset import ConcatDataset

from domainlab.tasks.a_task import NodeTaskDG
from domainlab.tasks.utils_task import (DsetDomainVecDecorator, mk_loader,
                                        mk_onehot)


class NodeTaskDict(NodeTaskDG):
    """
    Use dictionaries to create train and test domain split
    """
    def get_dset_by_domain(self, args, na_domain, split=False):
        """
        each domain correspond to one dataset, must be implemented by child class
        """
        raise NotImplementedError  # it is safe for each subclass to implement this

    def decorate_dset(self, model, args):
        """
        dispatch re-organization of data flow to model
        """

    def init_business(self, args, trainer=None):
        """
        create a dictionary of datasets
        """
        list_domain_tr, list_domain_te = self.get_list_domains_tr_te(args.tr_d, args.te_d)
        self.dict_dset_tr = {}
        self.dict_dset_val = {}
        dim_d = len(list_domain_tr)
        for (ind_domain_dummy, na_domain) in enumerate(list_domain_tr):
            dset_tr, dset_val = self.get_dset_by_domain(args, na_domain, split=args.split)
            vec_domain = mk_onehot(dim_d, ind_domain_dummy)  # for diva, dann
            ddset_tr = DsetDomainVecDecorator(dset_tr, vec_domain, na_domain)
            ddset_val = DsetDomainVecDecorator(dset_val, vec_domain, na_domain)
            if trainer is not None and hasattr(trainer, "dset_decoration_args_algo"):
                ddset_tr = trainer.dset_decoration_args_algo(args, ddset_tr)
                ddset_val = trainer.dset_decoration_args_algo(args, ddset_val)
            if trainer is not None and trainer.model is not None and hasattr(trainer.model, "dset_decoration_args_algo"):
                ddset_tr = trainer.model.dset_decoration_args_algo(args, ddset_tr)
                ddset_val = trainer.model.dset_decoration_args_algo(args, ddset_val)
            self.dict_dset_tr.update({na_domain: ddset_tr})
            self.dict_dset_val.update({na_domain: ddset_val})
        ddset_mix = ConcatDataset(tuple(self.dict_dset_tr.values()))
        self._loader_tr = mk_loader(ddset_mix, args.bs)

        ddset_mix_val = ConcatDataset(tuple(self.dict_dset_val.values()))
        self._loader_val = mk_loader(ddset_mix_val, args.bs,
                                     shuffle=False,
                                     drop_last=False)

        self.dict_dset_te = {}
        # No need to have domain Label for test
        for na_domain in list_domain_te:
            dset_te, *_ = self.get_dset_by_domain(args, na_domain, split=False)
            # NOTE: since get_dset_by_domain always return two datasets,
            # train and validation, this is not needed in test domain
            self.dict_dset_te.update({na_domain: dset_te})
        dset_te = ConcatDataset(tuple(self.dict_dset_te.values()))
        self._loader_te = mk_loader(dset_te, args.bs,
                                    shuffle=False,
                                    drop_last=False)