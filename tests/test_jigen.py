"""
end to end test, each file only test only 1 algorithm
so it is easier to identify which algorithm has a problem
"""
from domainlab.compos.exp.exp_main import Exp
from domainlab.arg_parser import mk_parser_main


def test_trainer_jigen():
    """
    end to end test
    """
    parser = mk_parser_main()
    margs = parser.parse_args(["--te_d", "caltech",
                               "--task", "mini_vlcs",
                               "--aname", "jigen", "--bs", "2",
                               "--epos", "20",
                               "--nname", "alexnet"])
    exp = Exp(margs)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
