"""
Use dictionaries to create train and test domain split
"""
from domainlab.tasks.b_task_classif import NodeTaskDictClassif  # abstract class


def mk_task_dset(dict_domain2dset,
                 list_str_y,
                 isize,
                 taskna,  # name of the task
                 dict_domain_img_trans=None,
                 img_trans_te=None,
                 parent=NodeTaskDictClassif,
                 succ=None):
    """
    make a task via a dictionary of dataset where the key is domain
    value is a tuple of dataset for training and dataset for
    validation (can be identical to training)
    """
    class NodeTaskDset(parent):
        """
        Use dictionaries to create train and test domain split
        """
        def conf(self, args):
            """
            set member variables
            """
            if dict_domain2dset is not None:
                self.dict_dset_all = dict_domain2dset
            self._name = taskna
            self._args = args  # for debug
            self.list_str_y = list_str_y
            self.isize = isize
            self._dict_domain_img_trans = dict_domain_img_trans
            self.img_trans_te = img_trans_te

        def get_dset_by_domain(self, args, na_domain, split=False):
            """
            each domain correspond to one dataset, must be implemented by child class
            """
            return self.dict_dset_all[na_domain]

        def init_business(self, args):
            """
            create a dictionary of datasets
            """
            self.conf(args)
            self.set_list_domains(list(dict_domain2dset.keys()))
            super().init_business(args)

        def add_tr_domain(self, name, dset_tr, trans=None, dset_val=None):
            """
            add domain via, name, dataset and transformations
            """
            self.dict_dset_all[name] = (dset_tr, dset_val)

    return NodeTaskDset(succ=succ)
