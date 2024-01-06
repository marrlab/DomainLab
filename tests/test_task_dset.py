"""
end to end test if the task constructed via dictionary of dsets works
"""
from tests.utils_test import utils_test_algo


def test_task_dset():
    """
    end to end test if the task constructed via dictionary of dsets works
    """
    args = "--te_d=0 --tpath=examples/tasks/task_dset_custom.py --debug --bs=2 \
        --nname=conv_bn_pool_2 --model=erm"
    utils_test_algo(args)
