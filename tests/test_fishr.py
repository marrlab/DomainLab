"""
unit and end-end test for deep all, mldg
"""
from tests.utils_test import utils_test_algo


def test_deepall_mldg():
    """
    train DeepAll with MLDG
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=2 --model=erm --trainer=fishr --nname=alexnet"
    utils_test_algo(args)

def test_fishr_resnet():
    """
    resnet
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=2 --model=erm --trainer=fishr --npath=examples/nets/resnet.py"
    utils_test_algo(args)
