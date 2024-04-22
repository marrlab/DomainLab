from domainlab.dsets.dset_mnist_color_solo_default import DsetMNISTColorSoloDefault
from domainlab.dsets.utils_data import DsetInMemDecorator


def test_dset_in_mem_decorator():
    dset = DsetMNISTColorSoloDefault(path="zdata", ind_color=1)
    dset_in_memory = DsetInMemDecorator(dset=dset)
    dset_in_memory.__len__()
    dset_in_memory.__getitem__(0)
