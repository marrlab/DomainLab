"""
unit and end-end test for hduva, matchdg
"""
from tests.utils_test import utils_test_algo


def test_hduva_matchdg():
    """
    train HDUVA with MATCH
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=2 --model=hduva --trainer=matchdg\
        --epochs_ctr=3 --epos=6 --nname=alexnet --gamma_y=7e5 \
        --nname_encoder_x2topic_h=conv_bn_pool_2 \
        --nname_encoder_sandwich_x2h4zd=alexnet"
    utils_test_algo(args)
