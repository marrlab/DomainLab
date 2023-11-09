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

def test_forcesetpoint_fbopt():
    """
    diva
    """
    args = "--te_d=0 --tr_d 1 2 --task=mnistcolor10 --bs=16 --aname=jigen --trainer=fbopt --nname=conv_bn_pool_2 --epos=10 --es=0 --mu_init=0.00001 --coeff_ma_setpoint=0.5 --coeff_ma_output_state=0.99 --force_setpoint_change_once"
    utils_test_algo(args)

def test_forcefeedforward_fbopt():
    args = "--te_d=0 --tr_d 1 2 --task=mnistcolor10 --bs=16 --aname=jigen --trainer=fbopt --nname=conv_bn_pool_2 --epos=2000 --epos_min=100 --es=1 --force_feedforward"
    utils_test_algo(args)
