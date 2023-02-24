"""
end to end test, each file only test only 1 algorithm
so it is easier to identify which algorithm has a problem
"""
from tests.utils_test import utils_test_algo


def test_mnist_color_jigen():
    """
    color minst on jigen
    """
    utils_test_algo("--te_d 0 1 2 --tr_d 3 7 --task=mnistcolor10 --aname=jigen \
                    --nname=conv_bn_pool_2")


def test_jigen30():
    """
    end to end test
    """
    utils_test_algo("--te_d 0 1 2 --tr_d 3 7 --task=mnistcolor10 --aname=jigen \
                    --nname=conv_bn_pool_2 --nperm=30")


def test_trainer_jigen100():
    """
    end to end test
    """
    utils_test_algo("--te_d 0 1 2 --tr_d 3 7 --task=mnistcolor10 --aname=jigen \
                    --nname=conv_bn_pool_2 --nperm=100")
