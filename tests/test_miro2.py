"""
end-end test for mutual information regulation
"""
from tests.utils_test import utils_test_algo


def test_miro2():
    """
    train with MIRO
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=2 --model=erm \
        --trainer=miro --nname=alexnet \
        --layers2extract_feats _net_invar_feat.net_torchvision.features.1"
    utils_test_algo(args)
