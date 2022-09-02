"""
merge several solo-color mnist to form a mixed dataset
"""
import numpy as np
from torch.utils.data import Dataset
from domainlab.dsets.dset_mnist_color_solo_default import DsetMNISTColorSoloDefault
from domainlab.dsets.utils_data import mk_fun_label2onehot
from domainlab.dsets.utils_data import plot_ds, plot_ds_list
from domainlab.dsets.dset_poly_domains_mnist_color_default import DsetMNISTColorMix
from domainlab.dsets.utils_data import DsetInMemDecorator


def test_DsetInMemDecorator():
    dset_in_memory = DsetInMemDecorator(None)