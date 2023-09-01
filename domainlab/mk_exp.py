"""
make an experiment
"""
from domainlab.arg_parser import mk_parser_main
from domainlab.compos.exp.exp_main import Exp
from domainlab.utils.utils_cuda import get_device
from domainlab.algos.msels.c_msel_val import MSelValPerf
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.b_obvisitor import ObVisitor



def mk_exp(task, model, trainer:str, test_domain:str, batchsize:int):
    """
    Creates a custom experiment. The user can specify the input parameters.

    Input Parameters:
        - task: create a task to a custom dataset by importing "mk_task_dset" function from
        "domainlab.tasks.task_dset". For more explanation on the input params refer to the
        documentation found in "domainlab.tasks.task_dset.py".
        - model: create a model [NameOfModel] by importing "mk_[NameOfModel]" function from
        "domainlab.models.model_[NameOfModel]". For a concrete example and explanation of the input
        params refer to the documentation found in "domainlab.models.model_[NameOfModel].py"
        - trainer: string,
        - test_domain: string,
        - batch size: int

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
