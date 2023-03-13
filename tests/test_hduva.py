"""
end to end test
"""

import gc
import torch
from domainlab.compos.exp.exp_main import Exp
from domainlab.arg_parser import mk_parser_main


def test_trainer_hduva():
    """
    end to end test
    """
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
    exp.trainer.tr_epoch(0)
    del exp
    torch.cuda.empty_cache()
    gc.collect()
