import os
from domainlab.compos.exp.exp_main import Exp
from domainlab.arg_parser import mk_parser_main


def test_deepall():
    parser = mk_parser_main()
    margs = parser.parse_args(["--te_d", "caltech",
                               "--task", "mini_vlcs",
                               "--aname", "deepall", "--bs", "2",
                               "--nname", "conv_bn_pool_2"
                               ])
    exp = Exp(margs)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)


def test_deepall_res():
    testdir = os.path.dirname(os.path.realpath(__file__))
    rootdir = os.path.join(testdir, "..")
    rootdir = os.path.abspath(rootdir)
    path = os.path.join(rootdir, "examples/nets/resnet.py")

    parser = mk_parser_main()
    margs = parser.parse_args(["--te_d", "caltech",
                               "--task", "mini_vlcs",
                               "--aname", "deepall", "--bs", "2",
                               "--npath", "%s" % (path)
                               ])
    exp = Exp(margs)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)



def test_dann():
    parser = mk_parser_main()
    margs = parser.parse_args(["--te_d", "caltech",
                               "--task", "mini_vlcs",
                               "--aname", "dann", "--bs", "2",
                               "--nname", "conv_bn_pool_2",
                               "--gamma_reg", "1.0"
                               ])
    exp = Exp(margs)
    exp.execute()
