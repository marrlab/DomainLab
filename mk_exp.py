"""
make an experiment
"""
from domainlab.arg_parser import mk_parser_main
from domainlab.compos.exp.exp_main import Exp

def mk_exp(task):
    parser = mk_parser_main()
    conf = parser.parse_args(str)
    exp = Exp(conf, task)
    return exp
