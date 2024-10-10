"""
loss ratio estimate
"""
from tests.utils_test import utils_test_algo


def test_loss_ratio_estimate():
    """
    test different number of batches
    """
    #args = "--te_d=0 --tr_d 2 4 --task=mnistcolor10 --debug --bs=8 \
    #    --model=erm --trainer=mldg --nname=conv_bn_pool_2"

    utils_test_algo(args)
