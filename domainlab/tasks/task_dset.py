"""
Use dictionaries to create train and test domain split
"""
from domainlab.tasks.b_task_classif import NodeTaskDictClassif  # abstract class


def mk_task_dset(isize,
                 taskna="task_custom",  # name of the task
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
            self._name = taskna

            if list_str_y is None and dim_y is None:
                raise RuntimeError("arguments list_str_y and dim_y can not be both None!")

            self.list_str_y = list_str_y   # list_str_y has to be initialized before dim_y
            self.dim_y = dim_y

            if self.list_str_y is None:
                self.list_str_y = [f"class{ele}" for ele in range(0, self.dim_y)]
            self.isize = isize

        def get_dset_by_domain(self, args, na_domain, split=False):
            """
            each domain correspond to one dataset, must be implemented by child class
            """
            return self.dict_dset_all[na_domain]

        def init_business(self, args, trainer=None):
            """
            create a dictionary of datasets
            """
            self.set_list_domains(list(self.dict_dset_all.keys()))
            super().init_business(args, trainer)

        def add_domain(self, name, dset_tr, dset_val=None):
            """
            add domain via, name, dataset and transformations
            """
            self.dict_dset_all[name] = (dset_tr, dset_val)
            # when add a new domain, change self state
            self.set_list_domains(list(self.dict_dset_all.keys()))

    return NodeTaskDset(succ=succ)
