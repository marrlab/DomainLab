"""
unit and end-end test for deep all, mldg
"""
from tests.utils_test import utils_test_algo


def test_deepall_mldg():
    """
    train DeepAll with MLDG
    """
    args = "-c examples/yaml/conf_diva_dial.yaml"
    utils_test_algo(args)
