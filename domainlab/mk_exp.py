"""
make an experiment
"""
from domainlab.arg_parser import mk_parser_main
from domainlab.compos.exp.exp_main import Exp
from domainlab.utils.utils_cuda import get_device
from domainlab.algos.msels.c_msel_val import MSelValPerf
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.b_obvisitor import ObVisitor



def mk_exp(task, model, trainer, test_domain, batchsize):
    """
    Creates a custom experiment. The user can specify the input parameters.

    Parameters: task, model, trainer, test_domain (must be a single string), batch size

    Returns: experiment
    """

    str_arg = f"--aname=apimodel --trainer={trainer} --te_d={test_domain} --bs={batchsize}"
    parser = mk_parser_main()
    conf = parser.parse_args(str_arg.split())
    device = get_device(conf)
    exp = Exp(conf, task, model=model)   # FIXME: trainer does not need to be executed twice
    model_sel = MSelOracleVisitor(MSelValPerf(max_es=conf.es))
    observer = ObVisitor(exp, model_sel, device)
    exp.trainer.init_business(model, task, observer, device, conf)
    return exp
