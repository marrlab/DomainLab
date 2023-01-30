"""
Test argparser functionality
"""

import sys
import pytest
from domainlab.arg_parser import parse_cmd_args


def test_parse_cmd_args_warning():
    """Call argparser for command line
    """
    sys.argv = ['main.py']
    with pytest.warns(Warning, match='no algorithm conf specified'):
        parse_cmd_args()


def test_parse_yml_args():
    """Test argparser with yaml file
    """

    #TODO: add tests
    pass