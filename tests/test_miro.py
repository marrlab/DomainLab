"""
end-end test for mutual information regulation
"""
from tests.utils_test import utils_test_algo


def test_miro():
    """
    train with MIRO
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=2 --model=erm \
        --trainer=miro --nname=alexnet --no_dump"
    utils_test_algo(args)
