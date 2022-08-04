import os
import torch
from domainlab.compos.exp.exp_main import Exp
from domainlab.arg_parser import mk_parser_main


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
