"""
end-end test
"""
from tests.utils_test import utils_test_algo


def test_irm():
    """
    train with Invariant Risk Minimization
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=2 --model=erm \
        --trainer=irm --nname=alexnet --no_dump"
    utils_test_algo(args)

def test_irm_sepdom():
    """
    train with Invariant Risk Minimization
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=2 --model=erm \
        --trainer=irmsepdom --nname=alexnet --no_dump"
    utils_test_algo(args)




def test_irm_scheduler():
    """
    train with Invariant Risk Minimization
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=2 --model=erm \
        --trainer=hyperscheduler_irm --nname=alexnet --no_dump"
    utils_test_algo(args)




def test_irm_mnist():
    """
    train with Invariant Risk Minimization
    """
    args = "--te_d=0 --task=mnistcolor10 --keep_model --model=erm \
        --trainer=irm --nname=conv_bn_pool_2 --no_dump"
    utils_test_algo(args)
