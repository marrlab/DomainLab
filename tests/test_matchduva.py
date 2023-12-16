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
        --nname_topic_distrib_img2topic=conv_bn_pool_2 \
        --nname_encoder_sandwich_layer_img2h4zd=alexnet"
    utils_test_algo(args)
