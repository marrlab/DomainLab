import torch
from domainlab.compos.exp.exp_main import Exp
from domainlab.arg_parser import mk_parser_main


def test_trainer_diva():
    parser = mk_parser_main()
    margs = parser.parse_args(["--te_d", "caltech",
                               "--task", "mini_vlcs",
                               "--aname", "matchdg", "--bs", "2",
                               "--nname", "conv_bn_pool_2",
                               "--epochs_ctr", "1",
                               "--epochs_erm", "1"])
    exp = Exp(margs)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
