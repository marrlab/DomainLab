"""
Color MNIST with palette
"""
from torch.utils.data import random_split

from domainlab.dsets.dset_mnist_color_solo_default import \
    DsetMNISTColorSoloDefault
from domainlab.dsets.utils_color_palette import default_rgb_palette  # @FIXME
from domainlab.tasks.b_task_classif import NodeTaskDictClassif
from domainlab.tasks.utils_task import ImSize
from domainlab.utils.utils_classif import mk_dummy_label_list_str


class NodeTaskMNISTColor10(NodeTaskDictClassif):
    """
    Use the deafult palette with 10 colors
    """
    @property
    def list_str_y(self):
        return mk_dummy_label_list_str("digit", 10)

    @property
    def isize(self):
        """image channel, height, width"""
        return ImSize(3, 28, 28)

    def get_list_domains(self):
        """
        1. get list of domain names
        2. better use method than property so new domains can be added
        """
        list_domains = []
        for rgb_list in default_rgb_palette:   # 10 colors
            domain = "_".join([str(c) for c in rgb_list])
            domain = "rgb_" + domain
            list_domains.append(domain)
        return list_domains

    def get_dset_by_domain(self, args, na_domain, split=True):  # @FIXME: different number of arguments than parent
        """get_dset_by_domain.
        :param args:
        :param na_domain:
        :param split: for test set, no need to split
        args.split: by default, split is set to be zero which in python can
        be evaluated in if statement, in which case, no validation set will be
        created. Otherwise, this argument is the split ratio
        """
        ratio_split = float(args.split) if split else False
        # by default, split is set to be zero which in python can
        # be evaluated in if statement, in which case, no validation
        # set will be created. Otherwise, this argument is
        # the split ratio
        ind_global = self.get_list_domains().index(na_domain)
        dset = DsetMNISTColorSoloDefault(ind_global, args.dpath)
        train_set = dset
        val_set = dset
        # split dset into training and test
        if ratio_split:
            train_len = int(len(dset) * ratio_split)
            val_len = len(dset) - train_len
            train_set, val_set = random_split(dset, [train_len, val_len])
        return train_set, val_set
