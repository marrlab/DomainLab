"""
make an experiment
"""
from domainlab.arg_parser import mk_parser_main
from domainlab.compos.exp.exp_main import Exp

def mk_exp(task, test_domain, batchsize):
    str_arg = f"--te_d={test_domain} --bs={batchsize}"
    parser = mk_parser_main()
    conf = parser.parse_args(str_arg.split())
    exp = Exp(conf, task)
    return exp
