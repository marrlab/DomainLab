import gc
import os

import pytest
import torch

from domainlab.arg_parser import mk_parser_main
from domainlab.exp.exp_main import Exp
from domainlab.models.model_custom import AModelCustom


def test_custom():
    testdir = os.path.dirname(os.path.realpath(__file__))
    rootdir = os.path.join(testdir, "..")
    rootdir = os.path.abspath(rootdir)
    mpath = os.path.join(rootdir, "examples/models/demo_custom_model.py")
    parser = mk_parser_main()
    argsstr = (
        "--te_d=caltech --task=mini_vlcs --model=custom --bs=2 --debug \
               --apath=%s --nname_argna2val my_custom_arg_name \
        --nname_argna2val alexnet --nname=alexnet"
        % (mpath)
    )
    margs = parser.parse_args(argsstr.split())
    exp = Exp(margs)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
    exp.trainer.post_tr()
    del exp
    torch.cuda.empty_cache()
    gc.collect()


def test_custom2():
    testdir = os.path.dirname(os.path.realpath(__file__))
    rootdir = os.path.join(testdir, "..")
    rootdir = os.path.abspath(rootdir)
    mpath = os.path.join(rootdir, "examples/models/demo_custom_model.py")
    path_net = os.path.join(rootdir, "examples/nets/resnet.py")
    parser = mk_parser_main()
    argsstr = (
        "--te_d=caltech --task=mini_vlcs --model=custom --bs=2 --debug \
               --apath=%s --npath_argna2val my_custom_arg_name \
        --npath_argna2val %s"
        % (mpath, path_net)
    )
    margs = parser.parse_args(argsstr.split())
    exp = Exp(margs)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
    exp.trainer.post_tr()
    del exp
    torch.cuda.empty_cache()
    gc.collect()


def test_no_network_exeption():
    """
    test if we can acess the exeption wen using a costum network
    which is not a network
    """
    parser = mk_parser_main()
    argsstr = "--te_d=caltech --task=mini_vlcs --debug \
              --bs=8 --model=erm --npath=tests/this_is_not_a_network.py"
    margs = parser.parse_args(argsstr.split())
    with pytest.raises(RuntimeError, match="the pytorch module returned by"):
        Exp(margs)
