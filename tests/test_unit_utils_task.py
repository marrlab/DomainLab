from domainlab.dsets.dset_poly_domains_mnist_color_default import DsetMNISTColorMix
from domainlab.tasks.utils_task import LoaderDomainLabel


def test_unit_utils_task():
    dset = DsetMNISTColorMix(n_domains=3, path="./output/")
    loader = LoaderDomainLabel(32, 3)(dset, 0, "0")
    batch = next(iter(loader))
    assert batch[0].shape == (32, 3, 28, 28)
