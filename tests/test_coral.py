"""
 end-end test
"""
from tests.utils_test import utils_test_algo


def test_coral():
    """
    coral
    """
    args = "--te_d 0 --tr_d 3 7 --bs=32 --debug --task=mnistcolor10 \
        --model=erm --nname=conv_bn_pool_2 --trainer=coral"
    utils_test_algo(args)
