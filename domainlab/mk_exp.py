"""
make an experiment
"""
from domainlab.arg_parser import mk_parser_main
from domainlab.exp.exp_main import Exp


def mk_exp(task, model, trainer: str, test_domain: str, batchsize: int, nocu=False):
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
    if nocu:
        str_arg += " --nocu"
    parser = mk_parser_main()
    conf = parser.parse_args(str_arg.split())
    exp = Exp(conf, task, model=model)
    return exp
