"""
 end-end test
"""
from tests.utils_test import utils_test_algo


def test_causal_irl():
    """
    causal irl
    """
    args = "--te_d 0 --tr_d 3 7 --bs=32 --debug --task=mnistcolor10 \
        --model=erm --nname=conv_bn_pool_2 --trainer=causalirl"
    utils_test_algo(args)
