import torch
from domainlab.compos.exp.exp_main import Exp
from domainlab.arg_parser import mk_parser_main


def test_trainer_diva():
    parser = mk_parser_main()
    margs = parser.parse_args(["--te_d", "caltech",
                               "--task", "mini_vlcs",
                               "--aname", "hduva", "--bs", "2",
                               "--nname", "conv_bn_pool_2",
                               "--gamma_y", "7e5",
                               "--nname_topic_distrib_img2topic", "conv_bn_pool_2",
                               "--nname_encoder_sandwich_layer_img2h4zd", "alexnet"
                               ])
    exp = Exp(margs)
    exp.trainer.before_tr()
    # exp.trainer.tr_epoch(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
