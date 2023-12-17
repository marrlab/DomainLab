"""
end to end test for trainer decorator
"""
from tests.utils_test import utils_test_algo


def test_cmd_model_decorator_diva():
    """
    trainer decorator
    """
    args = "--te_d=0 --task=mnistcolor10 --model=deepall,diva\
        --nname=conv_bn_pool_2 --nname_dom=conv_bn_pool_2 \
        --gamma_y=10e5 --gamma_d=1e5"
    utils_test_algo(args)

