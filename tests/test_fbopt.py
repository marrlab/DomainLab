"""
unit and end-end test for deep all, mldg
"""
from tests.utils_test import utils_test_algo


def test_dann_fbopt():
    """
    dann
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=2 --aname=dann --trainer=fbopt --nname=alexnet --epos=3"
    utils_test_algo(args)


def test_jigen_fbopt():
    """
    jigen
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=2 --aname=jigen --trainer=fbopt --nname=alexnet --epos=3"
    utils_test_algo(args)

def test_diva_fbopt():
    """
    diva
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=2 --aname=diva --gamma_y=1.0 --trainer=fbopt --nname=alexnet --epos=3"
    utils_test_algo(args)
