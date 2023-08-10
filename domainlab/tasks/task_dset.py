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
        @property
        def isize(self):
            return self._im_size

        @isize.setter
        def isize(self, isize):
            self._im_size = isize

        def conf(self, args):
            """
            set member variables
            """
            if dict_domain2dset is not None:
                self.dict_dset_tr = dict_domain2dset
            self._name = taskna
            self._args = args
            self.list_str_y = list_str_y
            self.isize = isize
            self._dict_domain_img_trans = dict_domain_img_trans
            self.img_trans_te = img_trans_te

        def init_business(self, args):
            """
            create a dictionary of datasets
            """
            self.conf(args)
            self.set_list_domains(list(dict_domain2dset.keys()))
            super().init_business(args)

        def add_domain(self, name, dset_tr, dset_val=None):
            self.dict_dset_tr[name] = dset_tr
            self.dict_dset_val = {}

    return NodeTaskDset(succ=succ)
