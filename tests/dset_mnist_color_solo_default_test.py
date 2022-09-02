import numpy as np
from domainlab.dsets.a_dset_mnist_color_rgb_solo import ADsetMNISTColorRGBSolo
from domainlab.dsets.utils_color_palette import default_rgb_palette
from domainlab.dsets.dset_mnist_color_solo_default import DsetMNISTColorSoloDefault
from domainlab.dsets.utils_data import plot_ds, plot_ds_list


def test_color_mnist():
    """
    test_color_mnist
    """
    dset = DsetMNISTColorSoloDefault(0, "zout")
    plot_ds(dset, "zout/color_solo.png")
    ds_list = []
    for i in range(10):
        ds_list.append(DsetMNISTColorSoloDefault(i, "zout"))
    plot_ds_list(ds_list, "zout/color_0_9.png")
