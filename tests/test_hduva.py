"""
end to end test
"""

import gc
import torch
from domainlab.exp.exp_main import Exp
from domainlab.arg_parser import mk_parser_main
from tests.utils_test import utils_test_algo

def test_hduva_zx_nonzero():
    """
    the combination of dial and diva: use dial trainer to train diva model
    """
    utils_test_algo("--te_d 0 1 2 --tr_d 3 7 --task=mnistcolor10 --zx_dim=8 \
                    --model=hduva --nname=conv_bn_pool_2 \
                    --nname_encoder_x2topic_h=conv_bn_pool_2 \
                    --gamma_y=7e5 \
                    --nname_encoder_sandwich_x2h4zd=conv_bn_pool_2")


def test_trainer_hduva():
    """
    end to end test
    """
    parser = mk_parser_main()
    margs = parser.parse_args(["--te_d", "caltech",
                               "--task", "mini_vlcs",
                               "--model", "hduva", "--bs", "2",
                               "--nname", "alexnet",
                               "--gamma_y", "7e5",
                               "--nname_encoder_x2topic_h", "conv_bn_pool_2",
                               "--nname_encoder_sandwich_x2h4zd", "conv_bn_pool_2"
                               ])
    exp = Exp(margs)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
    del exp
    torch.cuda.empty_cache()
    gc.collect()
