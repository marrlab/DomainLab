"""
unit and end-end test for deep all, dann
"""
import gc
import torch
from domainlab.exp.exp_main import Exp
from domainlab.arg_parser import mk_parser_main


def test_deepall():
    """
    unit deep all
    """
    parser = mk_parser_main()
    margs = parser.parse_args(["--te_d", "caltech",
                               "--task", "mini_vlcs",
                               "--aname", "deepall", "--bs", "2",
                               "--nname", "conv_bn_pool_2"
                               ])
    exp = Exp(margs)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
    exp.trainer.observer.update(True)
    del exp
    torch.cuda.empty_cache()
    gc.collect()


def test_deepall_trloss():
    """
    unit deep all
    """
    parser = mk_parser_main()
    margs = parser.parse_args(["--te_d", "caltech",
                               "--task", "mini_vlcs",
                               "--aname", "deepall", "--bs", "2",
                               "--nname", "conv_bn_pool_2",
                               "--msel", "loss_tr"
                               ])
    exp = Exp(margs)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
    exp.trainer.observer.update(True)
    del exp
    torch.cuda.empty_cache()
    gc.collect()
