"""
Test argparser functionality
"""
import os
from domainlab.arg_parser import mk_parser_main
from domainlab.exp.exp_main import Exp


def test_parse_yml_args():
    """Test argparser with yaml file"""
    testdir = os.path.dirname(os.path.realpath(__file__))
    rootdir = os.path.join(testdir, "..")
    rootdir = os.path.abspath(rootdir)
    file_path = os.path.join(rootdir, "examples/conf/vlcs_diva_mldg_dial.yaml")
    argsstr = "--config=" + file_path
    parser = mk_parser_main()
    margs = parser.parse_args(argsstr.split())
    exp = Exp(margs)
    exp.execute()
