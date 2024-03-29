"""
unit and end-end test for deep all, dann
"""
import gc
import os

import torch

from domainlab.arg_parser import mk_parser_main
from domainlab.exp.exp_main import Exp
from tests.utils_test import utils_test_algo


def test_erm():
    """
    unit deep all
    """
    parser = mk_parser_main()
    margs = parser.parse_args(
        [
            "--te_d",
            "caltech",
            "--task",
            "mini_vlcs",
            "--model",
            "erm",
            "--bs",
            "2",
            "--nname",
            "conv_bn_pool_2",
        ]
    )
    exp = Exp(margs)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
    del exp
    torch.cuda.empty_cache()
    gc.collect()


def test_erm_res():
    """
    resnet on deep all
    """
    testdir = os.path.dirname(os.path.realpath(__file__))
    rootdir = os.path.join(testdir, "..")
    rootdir = os.path.abspath(rootdir)
    path = os.path.join(rootdir, "examples/nets/resnet.py")

    parser = mk_parser_main()
    margs = parser.parse_args(
        [
            "--te_d",
            "caltech",
            "--task",
            "mini_vlcs",
            "--model",
            "erm",
            "--bs",
            "2",
            "--npath",
            f"{path}",
        ]
    )
    exp = Exp(margs)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
    del exp
    torch.cuda.empty_cache()
    gc.collect()


def test_erm_resdombed():
    """
    resnet on deep all
    """
    testdir = os.path.dirname(os.path.realpath(__file__))
    rootdir = os.path.join(testdir, "..")
    rootdir = os.path.abspath(rootdir)
    path = os.path.join(rootdir, "examples/nets/resnet50domainbed.py")

    parser = mk_parser_main()
    margs = parser.parse_args(
        [
            "--te_d",
            "caltech",
            "--task",
            "mini_vlcs",
            "--model",
            "erm",
            "--bs",
            "2",
            "--npath",
            f"{path}",
        ]
    )
    exp = Exp(margs)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
    del exp
    torch.cuda.empty_cache()
    gc.collect()


def test_dann():
    """
    domain adversarial neural network
    """
    parser = mk_parser_main()
    margs = parser.parse_args(
        [
            "--te_d",
            "caltech",
            "--task",
            "mini_vlcs",
            "--model",
            "dann",
            "--bs",
            "2",
            "--nname",
            "conv_bn_pool_2",
            "--gamma_reg",
            "1.0",
        ]
    )
    exp = Exp(margs)
    exp.execute()
    del exp
    torch.cuda.empty_cache()
    gc.collect()


def test_dann_dial():
    """
    train DANN with DIAL
    """
    args = "--te_d=caltech --task=mini_vlcs --model=dann --bs=2 --nname=alexnet --gamma_reg=1.0 --trainer=dial"
    utils_test_algo(args)


def test_sanity_check():
    """Sanity check of the dataset"""
    parser = mk_parser_main()
    margs = parser.parse_args(
        [
            "--te_d",
            "caltech",
            "--task",
            "mini_vlcs",
            "--model",
            "dann",
            "--bs",
            "2",
            "--nname",
            "conv_bn_pool_2",
            "--gamma_reg",
            "1.0",
            "--san_check",
            "--san_num",
            "4",
        ]
    )
    exp = Exp(margs)
    exp.execute()
    del exp
    torch.cuda.empty_cache()
    gc.collect()
