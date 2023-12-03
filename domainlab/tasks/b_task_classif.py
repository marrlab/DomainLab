"""
Use dictionaries to create train and test domain split
"""
import torch

from domainlab.tasks.a_task_classif import NodeTaskDGClassif
from domainlab.tasks.b_task import NodeTaskDict


class NodeTaskDictClassif(NodeTaskDict, NodeTaskDGClassif):
    """
    Use dictionaries to create train and test domain split
    """
    def init_business(self, args, trainer=None):
        """
        create a dictionary of datasets
        """
        super().init_business(args=args, trainer=trainer)
        self.count_domain_class()

    def count_domain_class(self):
        """
        iterate all domains and count the class label distribution for each
        return a double dictionary {"domain1": {"class1":3, "class2": 4,...}, ....}
        """
        for key, dset in self.dict_dset_tr.items():
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
        dict_class_count = {}
        for name, count in zip(self.list_str_y, list_count):
            dict_class_count[name] = count
        return dict_class_count
