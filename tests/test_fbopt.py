"""
unit and end-end test for deep all, mldg
"""
import pytest
from tests.utils_test import utils_test_algo


def test_dann_fbopt():
    """
    dann
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=2 --model=dann --trainer=fbopt --nname=alexnet --epos=3 --no_dump"
    utils_test_algo(args)


def test_jigen_fbopt():
    """
    jigen
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=2 --model=jigen --trainer=fbopt --nname=alexnet --epos=3 --no_dump"
    utils_test_algo(args)


def test_diva_fbopt():
    """
    diva
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=2 --model=diva --gamma_y=1.0 --trainer=fbopt --nname=alexnet --epos=3 --no_dump"
    utils_test_algo(args)
