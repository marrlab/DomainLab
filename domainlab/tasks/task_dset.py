"""
Use dictionaries to create train and test domain split
"""
from domainlab.tasks.b_task_classif import NodeTaskDictClassif  # abstract class


def mk_task_dset(isize,
                 taskna="task_custom",  # name of the task
                 dict_domain2dset=None,
                 dim_y=None,
                 list_str_y=None,
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
        def conf_without_args(self):
            """
            set member variables
            """
            if dict_domain2dset is not None:
                self.dict_dset_all = dict_domain2dset
            self._name = taskna
            self.dim_y = dim_y
            self.list_str_y = list_str_y
            if self.list_str_y is None and self.dim_y is None:
                raise RuntimeError("list_str_y and dim_y can not be both None!")
            if self.list_str_y is None:
                self.list_str_y=[f"class{ele}" for ele in range(0, self.dim_y)]
            self.isize = isize

        def get_dset_by_domain(self, args, na_domain, split=False):
            """
            each domain correspond to one dataset, must be implemented by child class
            """
            return self.dict_dset_all[na_domain]

        def init_business(self, args):
            """
            create a dictionary of datasets
            """
            self._args = args  # for debug
            self.set_list_domains(list(self.dict_dset_all.keys()))
            super().init_business(args)

        def add_domain(self, name, dset_tr, dset_val=None):
            """
            add domain via, name, dataset and transformations
            """
            self.dict_dset_all[name] = (dset_tr, dset_val)

    return NodeTaskDset(succ=succ)
