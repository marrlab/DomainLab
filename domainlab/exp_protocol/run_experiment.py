"""
Runs one task for a single hyperparameter sample for each leave-out-domain
and each random seed.
"""
import gc
import ast

import pandas as pd
import torch

from domainlab.arg_parser import mk_parser_main
from domainlab.compos.exp.exp_cuda_seed import set_seed
from domainlab.compos.exp.exp_main import Exp
from domainlab.compos.exp.exp_utils import ExpProtocolAggWriter


def load_parameters(file: str, index: int) -> tuple:
    """
    Loads a single parameter sample
    @param file: csv file
    @param index: index of hyper-parameter
    """
    param_df = pd.read_csv(file, index_col=0)
    row = param_df.loc[index]
    params = ast.literal_eval(row.params)
    return row.task, params


def apply_dict_to_args(args, data: dict, extend=False):
    """
    Tries to apply the data to the args dict of DomainLab.
    Unknown keys are silently ignored as long as
    extend is not set.
    # FIXME: do we have a test to ensure args dict from
    # domainlab really got what is passed from "data" dict?
    """
    arg_dict = args.__dict__
    for key, value in data.items():
        if key in arg_dict or extend:
            if isinstance(value, list):
                cur_val = arg_dict.get(key, None)
                if not isinstance(cur_val, list):
                    arg_dict[key] = []
                arg_dict[key].extend(value)
            else:
                arg_dict[key] = value


def run_experiment(
        config: dict,
        param_file: str,
        param_index: int,
        out_file: str,
        misc=None,
):
    """
    Runs the experiment several times:

    for test_domain in test_domains:
        for seed from startseed to endseed:
            evaluate the algorithm with test_domain, initialization with seed

    :param config: dictionary from the benchmark yaml
    :param param_file: path to the csv with the parameter samples
    :param param_index: parameter index that should be covered by this task,
    currently this correspond to the line number in the csv file, or row number
    in the resulting pandas dataframe
    :param out_file: path to the output csv
    :param misc: optional dictionary of additional parameters, if any.

    # FIXME: we might want to run the experiment using commandline arguments
    """
    if misc is None:
        misc = {}
    task, hyperparameters = load_parameters(param_file, param_index)
    # print("\n*******************************************************************")
    # print(f"{task}, param_index={param_index}, params={hyperparameters}")
    # print("*******************************************************************\n")
    misc['result_file'] = out_file
    misc['params'] = hyperparameters
    misc['benchmark_task_name'] = task
    misc['param_index'] = param_index
    misc['keep_model'] = False
    misc['no_dump'] = True

    parser = mk_parser_main()
    args = parser.parse_args(args=[])
    apply_dict_to_args(args, config)
    apply_dict_to_args(args, config[task])
    apply_dict_to_args(args, hyperparameters)
    apply_dict_to_args(args, misc, extend=True)

    for te_d in config['test_domains']:
        args.te_d = te_d
        for seed in range(config['startseed'], config['endseed'] + 1):
            set_seed(seed)
            args.seed = seed
            print(torch.cuda.memory_summary())
            exp = Exp(args=args, visitor=ExpProtocolAggWriter)
            if not misc.get('testing', False):
                exp.execute()
            del exp
            torch.cuda.empty_cache()
            gc.collect()
            try:
                if torch.cuda.is_available():
                    print(torch.cuda.memory_summary())
            except KeyError as ex:
                print(ex)
