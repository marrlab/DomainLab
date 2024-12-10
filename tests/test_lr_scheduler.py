
"""
unit and end-end test for  lr scheduler
"""
from tests.utils_test import utils_test_algo


def test_lr_scheduler():
    """
    train
    """
    args = "--te_d=2 --tr_d 0 1 --task=mnistcolor10 --debug --bs=100 --model=erm \
        --nname=conv_bn_pool_2 --no_dump --lr_scheduler=CosineAnnealingLR"
    utils_test_algo(args)
