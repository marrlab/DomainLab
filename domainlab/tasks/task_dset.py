"""
Use dictionaries to create train and test domain split
"""
from domainlab.tasks.b_task_classif import NodeTaskDictClassif


def mk_task_dset(dict_domain2dset,
                 list_str_y,
                 isize,
                 taskna,  # name of the task
                 dict_domain_img_trans=None,
                 img_trans_te=None,
                 parent=NodeTaskDictClassif,
                 succ=None):
    class NodeTaskDset(parent):
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
            self._dict_domain_img_trans = dict_domain_img_trans
            self.img_trans_te = img_trans_te

        def get_dset_by_domain(self, args, na_domain, split=None):
            return dict_domain2dset[na_domain]

        def init_business(self, args):
            """
            create a dictionary of datasets
            """
            self.conf(args)
            self.set_list_domains(list(dict_domain2dset.keys()))
            super().init_business(args)
    return NodeTaskDset(succ=succ)
