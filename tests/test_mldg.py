"""
unit and end-end test for deep all, mldg
"""
from tests.utils_test import utils_test_algo


def test_erm_mldg():
    """
    train DeepAll with MLDG
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=8 --model=erm --trainer=mldg --nname=alexnet"
    utils_test_algo(args)
