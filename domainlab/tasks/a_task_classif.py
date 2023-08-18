"""
Abstract class for TaskClassif
"""
import os

from domainlab.tasks.utils_task import img_loader2dir
from domainlab.tasks.a_task import NodeTaskDG


class NodeTaskDGClassif(NodeTaskDG):
    """
    abstract class for classification task
    """
    def __init__(self, succ=None):
        # just for declaration of variables
        self._list_str_y = None
        self._dim_y = None
        # super() must come last instead of in the beginning
        super().__init__(succ)

    @property
    def list_str_y(self):
        """
        getter for list_str_y
        """
        return self._list_str_y

    @list_str_y.setter
    def list_str_y(self, list_str_y):
        """
        setter for list_str_y
        """
        self._list_str_y = list_str_y

    @property
    def dim_y(self):
        """classification dimension"""
        if self._dim_y is None:
            return len(self.list_str_y)
        return self._dim_y

    def sample_sav(self, root, batches=5, subfolder_na="task_sample"):
        """
        sample data from task and save to disk
        """
        folder_na = os.path.join(root, self.task_name, subfolder_na)

        img_loader2dir(self.loader_te,
                       list_domain_na=self.get_list_domains(),
                       list_class_na=self.list_str_y,
                       folder=folder_na,
                       batches=batches,
                       test=True)

        img_loader2dir(self.loader_tr,
                       list_domain_na=self.get_list_domains(),
                       list_class_na=self.list_str_y,
                       folder=folder_na,
                       batches=batches)
