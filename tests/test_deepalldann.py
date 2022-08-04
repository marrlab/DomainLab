import torch
from domainlab.compos.exp.exp_main import Exp
from domainlab.arg_parser import mk_parser_main


def test_deepall():
    parser = mk_parser_main()
    margs = parser.parse_args(["--te_d", "caltech",
                               "--task", "mini_vlcs",
                               "--aname", "deepall", "--bs", "2",
                               "--nname", "conv_bn_pool_2"
                               ])
    exp = Exp(margs)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()


def test_dann():
    parser = mk_parser_main()
    margs = parser.parse_args(["--te_d", "caltech",
                               "--task", "mini_vlcs",
                               "--aname", "dann", "--bs", "2",
                               "--nname", "conv_bn_pool_2",
                               "--gamma_reg", "1.0"
                               ])
    exp = Exp(margs)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
