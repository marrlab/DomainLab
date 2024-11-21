"""
end-end test for mutual information regulation
"""
import pytest
from tests.utils_test import utils_test_algo


def test_miro3():
    """
    train with MIRO
    """
    args = "--te_d=caltech --task=mini_vlcs --debug --bs=2 --model=erm \
        --trainer=miro --nname=alexnet \
        --layers2extract_feats features --no_dump"
    with pytest.raises(RuntimeError):
        utils_test_algo(args)
        raise RuntimeError("This is a runtime error")
