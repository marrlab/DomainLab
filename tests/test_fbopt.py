"""
unit and end-end test for deep all, mldg
"""
from tests.utils_test import utils_test_algo


def test_deepall_fbopt():
    """
    train DeepAll with MLDG
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=2 --aname=dann --trainer=fbopt --nname=alexnet --epos=3"
    utils_test_algo(args)
