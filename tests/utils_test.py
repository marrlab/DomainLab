from domainlab.compos.exp.exp_main import Exp
from domainlab.arg_parser import mk_parser_main


def utils_test_algo(argsstr="--help"):
    parser = mk_parser_main()
    margs = parser.parse_args(argsstr.split())
    exp = Exp(margs)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
    exp.trainer.post_tr()
