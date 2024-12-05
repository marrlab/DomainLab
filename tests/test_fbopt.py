"""
unit and end-end test for deep all, mldg
"""
import pytest
from tests.utils_test import utils_test_algo


def test_dann_fbopt():
    """
    dann
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=2 --model=dann --trainer=fbopt --nname=alexnet --epos=3"
    utils_test_algo(args)


def test_jigen_fbopt():
    """
    jigen
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=2 --model=jigen --trainer=fbopt --nname=alexnet --epos=3"
    utils_test_algo(args)


def test_diva_fbopt():
    """
    diva
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=2 --model=diva --gamma_y=1.0 --trainer=fbopt --nname=alexnet --epos=3"
    utils_test_algo(args)


def test_erm_fbopt():
    """
    erm
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=2 --model=erm --trainer=fbopt --nname=alexnet --epos=3" # pylint: disable=line-too-long
    with pytest.raises(RuntimeError):
        utils_test_algo(args)


def test_irm_fbopt():
    """
    irm
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=2 --model=erm --trainer=fbopt_irm --nname=alexnet --epos=3" # pylint: disable=line-too-long
    utils_test_algo(args)


def test_forcesetpoint_fbopt():
    """
    diva
    """
    args = "--te_d=0 --tr_d 1 2 --task=mnistcolor10 --bs=16 --model=jigen --trainer=fbopt --nname=conv_bn_pool_2 --epos=10 --es=0 --mu_init=0.00001 --coeff_ma_setpoint=0.5 --coeff_ma_output_state=0.99 --force_setpoint_change_once"
    utils_test_algo(args)
