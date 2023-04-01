"""
unit and end-end test for diva, dial
"""
from tests.utils_test import utils_test_algo


def test_diva_dial():
    """
    train diva with dial
    """
    args = "-c examples/yaml/conf_diva_dial.yaml"
    utils_test_algo(args)
