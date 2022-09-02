from domainlab.dsets.dset_mnist_color_solo_default import DsetMNISTColorSoloDefault
from domainlab.dsets.utils_data import DsetInMemDecorator


def test_DsetInMemDecorator():
    dset_in_memory = DsetInMemDecorator(dset=DsetMNISTColorSoloDefault( path = "../data", 
                                                                        ind_color=1))
    dset_in_memory.__len__()
    dset_in_memory.__getitem__(0)