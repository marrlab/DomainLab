"""
 end-end test
"""
from tests.utils_test import utils_test_algo


def test_mhof_irm():
    """
    mhof-irm
    """
    args = "--te_d=0 --task=mnistcolor10 --model=erm \
        --trainer=fbopt_irm --nname=conv_bn_pool_2 \
        --k_i_gain_ratio=0.5"
    utils_test_algo(args)
