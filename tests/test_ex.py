from tests.utils_test import utils_test_algo

def test_example():
    utils_test_algo("--te_d=0 --task=mnistcolor10 --keep_model --aname=diva --nname=conv_bn_pool_2 --nname_dom=conv_bn_pool_2 --gamma_y=10e5 --gamma_d=1e5")
