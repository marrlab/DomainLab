"""
loss ratio estimate
"""
from tests.utils_test import utils_test_algo


def test_loss_ratio_estimate0():
    """
    test different number of batches
    """
    args = "--te_d=0 --tr_d 2 4 --task=mnistcolor10 --debug --bs=8 \
        --model=erm --trainer=mldg --nname=conv_bn_pool_2 \
        --nb4reg_over_task_ratio=0"
    utils_test_algo(args)


def test_loss_ratio_estimate100():
    """
    test different number of batches
    """
    args = "--te_d=0 --tr_d 2 4 --task=mnistcolor10 --debug --bs=8 \
        --model=erm --trainer=mldg --nname=conv_bn_pool_2 \
        --nb4reg_over_task_ratio=100"
    utils_test_algo(args)
