from domainlab.dsets.dset_mnist_color_solo_default import DsetMNISTColorSoloDefault
from domainlab.dsets.utils_data import plot_ds, plot_ds_list


def test_color_mnist():
    """
    test_color_mnist
    """
    dset = DsetMNISTColorSoloDefault(0, "zoutput")
    plot_ds(dset, "zoutput/color_solo.png")
    ds_list = []
    for i in range(10):
        ds_list.append(DsetMNISTColorSoloDefault(i, "zoutput"))
    plot_ds_list(ds_list, "zoutput/color_0_9.png")


def test_color_mnist2():
    """
    test_color_mnist
    """
    dset = DsetMNISTColorSoloDefault(0, "zoutput", flag_rand_color=True)
    plot_ds(dset, "zoutput/color_solo.png")
    ds_list = []
    for i in range(10):
        ds_list.append(DsetMNISTColorSoloDefault(i, "zoutput"))
    plot_ds_list(ds_list, "zoutput/color_0_9.png")


def test_color_mnist3():
    """
    test_color_mnist
    """
    dset = DsetMNISTColorSoloDefault(0, "zoutput", color_scheme="num", raw_split="test")
    plot_ds(dset, "zoutput/color_solo.png")
    ds_list = []
    for i in range(10):
        ds_list.append(DsetMNISTColorSoloDefault(i, "zoutput"))
    plot_ds_list(ds_list, "zoutput/color_0_9.png")


def test_color_mnist4():
    """
    test_color_mnist
    """
    dset = DsetMNISTColorSoloDefault(0, "zoutput", color_scheme="back", raw_split="test")
    plot_ds(dset, "zoutput/color_solo.png")
    ds_list = []
    for i in range(10):
        ds_list.append(DsetMNISTColorSoloDefault(i, "zoutput"))
    plot_ds_list(ds_list, "zoutput/color_0_9.png")
