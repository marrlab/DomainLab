"""
for end to end test
helper function to execute as if command line arguments are passed
"""
import gc

import pandas as pd
import torch

from domainlab.arg_parser import mk_parser_main
from domainlab.exp.exp_main import Exp


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
    ret = shell.run("rm", "-rf zoutput")
    assert ret.returncode == 0


def assert_frame_not_equal(*args, **kwargs):
    """
    use try except to assert frame not equal in pandas
    """
    try:
        pd.testing.assert_frame_equal(*args, **kwargs)
    except AssertionError:
        # frames are not equal
        pass
    else:
        # frames are equal
        raise AssertionError
