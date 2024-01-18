"""
merge several solo-color mnist to form a mixed dataset
"""

from domainlab.dsets.dset_poly_domains_mnist_color_default import DsetMNISTColorMix
from domainlab.dsets.utils_data import plot_ds


def test_color_mnist():
    dset = DsetMNISTColorMix(n_domains=3, path="./output/")
    plot_ds(dset, "color_mix.png")
