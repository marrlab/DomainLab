"""
end to end test, each file only test only 1 algorithm
so it is easier to identify which algorithm has a problem
"""
import gc

import torch

from domainlab.arg_parser import mk_parser_main
from domainlab.exp.exp_main import Exp


def test_trainer_dial():
    """
    end to end test
    """
    parser = mk_parser_main()
    margs = parser.parse_args(
        [
            "--te_d",
            "0",
            "--task",
            "mnistcolor10",
            "--model",
            "erm",
            "--bs",
            "2",
            "--trainer",
            "dial",
            "--nname",
            "conv_bn_pool_2",
        ]
    )
    exp = Exp(margs)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
    del exp
    torch.cuda.empty_cache()
    gc.collect()
