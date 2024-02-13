"""
unit and end-end test for deep all, dann
"""
from tests.utils_test import utils_test_algo


def test_erm():
    """
    unit deep all
    """
    utils_test_algo(
        "--te_d 0 --tr_d 3 7 --task=mnistcolor10 \
                    --model=erm --nname=conv_bn_pool_2 --bs=2 \
                    --msel=loss_tr --epos=2"
    )
