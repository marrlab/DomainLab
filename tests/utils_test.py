"""
for end to end test
helper function to execute as if command line arguments are passed
"""
import gc
import torch
from domainlab.compos.exp.exp_main import Exp
from domainlab.arg_parser import mk_parser_main


def utils_test_algo(argsstr="--help"):
    """
    helper function to execute as if command line arguments are passed
    """
    parser = mk_parser_main()
    margs = parser.parse_args(argsstr.split())
    exp = Exp(margs)
    exp.execute()
    del exp
    torch.cuda.empty_cache()
    gc.collect()
