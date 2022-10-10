"""
merge several solo-color mnist to form a mixed dataset
"""
import numpy as np
from torch.utils.data import Dataset

from domainlab.dsets.dset_mnist_color_solo_default import \
    DsetMNISTColorSoloDefault
from domainlab.dsets.utils_data import mk_fun_label2onehot


class DsetMNISTColorMix(Dataset):
    """
    merge several solo-color mnist to form a mixed dataset
    """
    def __init__(self, n_domains, path, color_scheme='both'):
        self.n_domains = n_domains
        self.list_dset = [None] * n_domains
        self.fun_dlabel2onehot = mk_fun_label2onehot(n_domains)
        for domain_ind in range(n_domains):
            self.list_dset[domain_ind] = \
                DsetMNISTColorSoloDefault(domain_ind, path,
                                          color_scheme=color_scheme)
        self.list_len = [len(ds) for ds in self.list_dset]
        self.size_single = min(self.list_len)

    def __len__(self):
        """__len__."""
        return sum(self.list_len)

    def __getitem__(self, idx):
        rand_domain = np.random.random_integers(self.n_domains-1)  # @FIXME
        idx_local = idx % self.size_single
        img, c_label = self.list_dset[rand_domain][idx_local]
        return img, c_label, self.fun_dlabel2onehot(rand_domain)


class DsetMNISTColorMixNoDomainLabel(DsetMNISTColorMix):
    """
    DsetMNISTColorMixNoDomainLabel
    """
    def __getitem__(self, idx):
        img, c_label, _ = super().__getitem__(idx)
        return img, c_label
