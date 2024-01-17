import gc

import torch

from domainlab.arg_parser import mk_parser_main
from domainlab.exp.exp_main import Exp


def test_trainer_matchdg():
    parser = mk_parser_main()
    margs = parser.parse_args(
        [
            "--te_d",
            "caltech",
            "--task",
            "mini_vlcs",
            "--trainer",
            "matchdg",
            "--bs",
            "2",
            "--model",
            "erm",
            "--nname",
            "conv_bn_pool_2",
            "--epochs_ctr",
            "1",
            "--epos",
            "3",
        ]
    )
    exp = Exp(margs)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
    del exp
    torch.cuda.empty_cache()
    gc.collect()
