"""
end to end test for trainer decorator
"""
from tests.utils_test import utils_test_algo


def test_trainer_decorator():
    """
    trainer decorator
    """
    args = "--te_d=0 --task=mnistcolor10 --keep_model --aname=diva \
        --nname=conv_bn_pool_2 --nname_dom=conv_bn_pool_2 \
        --gamma_y=10e5 --gamma_d=1e5 --trainer=dial,mldg"
    utils_test_algo(args)

def test_trainer_decorator2():
    """
    trainer decorator
    """
    args = "--te_d=0 --task=mnistcolor10 --keep_model --aname=diva \
        --nname=conv_bn_pool_2 --nname_dom=conv_bn_pool_2 \
        --gamma_y=10e5 --gamma_d=1e5 --trainer=mldg,dial"
    utils_test_algo(args)
