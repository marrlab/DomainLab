from domainlab.tasks.utils_task import LoaderDomainLabel
from domainlab.dsets.dset_poly_domains_mnist_color_default import DsetMNISTColorMix


def test_unit_utils_task():
    dset = DsetMNISTColorMix(n_domains=3, path="./output/")
    ld = LoaderDomainLabel(32, 3)
    ld(dset, 0, "0")

