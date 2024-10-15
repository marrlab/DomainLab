"""
end-end test for mutual information regulation
"""
from tests.utils_test import utils_test_algo


def test_miro2():
    """
    train with MIRO
    """
    args = "--te_d=2 --tr_d 0 1 --task=mnistcolor10 --debug --bs=100 --model=erm \
        --trainer=miro --nname=conv_bn_pool_2 \
        --layers2extract_feats _net_invar_feat.conv_net.5"
    utils_test_algo(args)
