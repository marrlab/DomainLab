import pytest
import sys
from domainlab.arg_parser import parse_cmd_args


def test_parse_cmd_args_warning():
    sys.argv = ['main.py']
    with pytest.warns(Warning, match='no algorithm conf specified'):
        parse_cmd_args()