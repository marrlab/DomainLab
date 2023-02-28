import os
import gc
import torch
from domainlab.compos.exp.exp_main import Exp
from domainlab.arg_parser import mk_parser_main
from tests.utils_test import utils_test_algo


def test_dial_diva():
    """
    the combination of dial and diva: use dial trainer to train diva model
    """
    utils_test_algo("--te_d 0 1 2 --tr_d 3 7 --task=mnistcolor10 --aname=diva \
                    --nname=conv_bn_pool_2 --nname_dom=conv_bn_pool_2 \
                    --gamma_y=7e5 --gamma_d=1e5 --trainer=dial")

def test_diva():
    parser = mk_parser_main()
    argsstr = "--te_d=caltech --task=mini_vlcs --aname=diva --bs=2 \
               --nname=conv_bn_pool_2 --gamma_y=7e5 --gamma_d=7e5 \
               --nname_dom=conv_bn_pool_2 --gen --nocu"
    margs = parser.parse_args(argsstr.split())
    exp = Exp(margs)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
    exp.trainer.post_tr()
    del exp
    torch.cuda.empty_cache()
    gc.collect()



def test_trainer_diva():
    parser = mk_parser_main()
    margs = parser.parse_args(["--te_d", "caltech",
                               "--task", "mini_vlcs",
                               "--aname", "diva", "--bs", "2",
                               "--nname", "conv_bn_pool_2",
                               "--gamma_y", "7e5",
                               "--gamma_d", "7e5",
                               "--nname_dom", "conv_bn_pool_2"
                               ])
    exp = Exp(margs)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
    del exp
    torch.cuda.empty_cache()
    gc.collect()



def test_trainer_diva_folder():
    testdir = os.path.dirname(os.path.realpath(__file__))
    rootdir = os.path.join(testdir, "..")
    rootdir = os.path.abspath(rootdir)
    path = os.path.join(rootdir, "examples/tasks/task_vlcs.py")
    parser = mk_parser_main()
    margs = parser.parse_args(["--te_d", "caltech",
                               "--tpath", "%s" % (path),
                               "--aname", "diva", "--bs", "2",
                               "--nname", "conv_bn_pool_2",
                               "--gamma_y", "7e5",
                               "--gamma_d", "7e5",
                               "--nname_dom", "conv_bn_pool_2"
                               ])
    exp = Exp(margs)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
    del exp
    torch.cuda.empty_cache()
    gc.collect()



def test_trainer_diva_pathlist():
    testdir = os.path.dirname(os.path.realpath(__file__))
    rootdir = os.path.join(testdir, "..")
    rootdir = os.path.abspath(rootdir)
    path = os.path.join(rootdir, "examples/tasks/demo_task_path_list_small.py")
    parser = mk_parser_main()
    margs = parser.parse_args(["--te_d", "sketch",
                               "--tpath", "%s" % (path),
                               "--aname", "diva", "--bs", "2",
                               "--nname", "conv_bn_pool_2",
                               "--gamma_y", "7e5",
                               "--gamma_d", "7e5",
                               "--nname_dom", "conv_bn_pool_2"
                               ])
    exp = Exp(margs)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
    del exp
    torch.cuda.empty_cache()
    gc.collect()

