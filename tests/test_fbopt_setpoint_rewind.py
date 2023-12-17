"""
unit and end-end test for deep all, mldg
"""
from tests.utils_test import utils_test_algo

def test_jigen_fbopt():
    """
    jigen
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=2 --model=jigen --trainer=fbopt --nname=alexnet --epos=300 --setpoint_rewind=yes"
    utils_test_algo(args)
