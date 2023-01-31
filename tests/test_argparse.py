"""
Test argparser functionality
"""

import sys
import os
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
    testdir = os.path.dirname(os.path.realpath(__file__))
    rootdir = os.path.join(testdir, "..")
    rootdir = os.path.abspath(rootdir)
    file_path = os.path.join(rootdir, "examples/yaml/demo.yaml")
    sys.argv = ['main.py', '--config=' + file_path]
    args = parse_cmd_args()

    # Checking if arguments are from demo.yaml
    assert args.te_d == "caltech"
    assert args.tpath == "examples/tasks/task_vlcs.py"
    assert args.bs == 2
    assert args.aname == "diva"
    assert args.gamma_y == 700000.0