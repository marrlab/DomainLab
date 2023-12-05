"""
Runs one task for a single hyperparameter sample for each leave-out-domain
and each random seed.
"""
import ast
import gc
import copy
import numpy as np
import pandas as pd
import torch

from domainlab.arg_parser import mk_parser_main, apply_dict_to_args
from domainlab.exp.exp_cuda_seed import set_seed
from domainlab.exp.exp_main import Exp
from domainlab.exp.exp_utils import ExpProtocolAggWriter
from domainlab.utils.logger import Logger


def load_parameters(file: str, index: int) -> tuple:
    """
    Loads a single parameter sample
    @param file: csv file
    @param index: index of hyper-parameter
    """
    param_df = pd.read_csv(file, index_col=0)
    row = param_df.loc[index]
    params = ast.literal_eval(row.params)
    # row.task has nothing to do with DomainLab task, it is
    # benchmark task which correspond to one algorithm
    return row.task, params


def convert_dict2float(dict_in):
    """
    convert scientific notation from 1e5 to 10000
    """
    dict_out = copy.deepcopy(dict_in)
    for key, val in dict_out.items():
        if isinstance(val, str):
            try:
                val_float = float(val)
                dict_out[key] = val_float
            except:
                pass
    return dict_out


def run_experiment(
        config: dict,
        param_file: str,
        param_index: int,
        out_file: str,
        start_seed=None,
        misc=None,
        num_gpus=1
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
    :param start_seed: random seed to start for stochastic variations of pytorch
    :param misc: optional dictionary of additional parameters, if any.

    # FIXME: we might want to run the experiment using commandline arguments
    """

    if misc is None:
        misc = {}
    str_algo_as_task, hyperparameters = load_parameters(param_file, param_index)
    logger = Logger.get_logger()
    logger.debug("\n*******************************************************************")
    logger.debug(f"{str_algo_as_task}, param_index={param_index}, params={hyperparameters}")
    logger.debug("*******************************************************************\n")
    misc['result_file'] = out_file
    misc['params'] = hyperparameters
    misc['benchmark_task_name'] = str_algo_as_task
    misc['param_index'] = param_index
    misc['keep_model'] = False

    parser = mk_parser_main()
    args = parser.parse_args(args=[])
    args_algo_specific = config[str_algo_as_task].copy()
    if 'hyperparameters' in args_algo_specific:
        del args_algo_specific['hyperparameters']
    args_domainlab_common_raw = config.get("domainlab_args", {})
    args_domainlab_common = convert_dict2float(args_domainlab_common_raw)
    # check if some of the hyperparameters are already specified
    # in args_domainlab_common or args_algo_specific
    if np.intersect1d(list(args_algo_specific.keys()),
                      list(hyperparameters.keys())).shape[0] > 0:
        logger.error(f"the hyperparameter "
                  f"{np.intersect1d(list(args_algo_specific.keys()), list(hyperparameters.keys()))}"
                  f" has already been fixed to a value in the algorithm section.")
        raise RuntimeError(f"the hyperparameter "
                  f"{np.intersect1d(list(args_algo_specific.keys()), list(hyperparameters.keys()))}"
                  f" has already been fixed to a value in the algorithm section.")
    if np.intersect1d(list(args_domainlab_common.keys()),
                      list(hyperparameters.keys())).shape[0] > 0:
        logger.error(f"the hyperparameter "
                  f"{np.intersect1d(list(args_algo_specific.keys()), list(hyperparameters.keys()))}"
                  f" has already been fixed to a value in the domainlab_args section.")
        raise RuntimeError(f"the hyperparameter "
                  f"{np.intersect1d(list(args_algo_specific.keys()), list(hyperparameters.keys()))}"
                  f" has already been fixed to a value in the domainlab_args section.")
    apply_dict_to_args(args, args_domainlab_common)
    args_algo_specific_scientific_notation = convert_dict2float(args_algo_specific)
    apply_dict_to_args(args, args_algo_specific_scientific_notation, extend=True)
    apply_dict_to_args(args, hyperparameters)
    apply_dict_to_args(args, misc, extend=True)
    gpu_ind = param_index % num_gpus
    args.device = str(gpu_ind)

    if torch.cuda.is_available():
        torch.cuda.init()
        logger.info("before experiment loop: ")
        logger.info(str(torch.cuda.memory_summary()))
    if start_seed is None:
        start_seed = config['startseed']
        end_seed = config['endseed']
    else:
        end_seed = start_seed + (config['endseed'] - config['startseed'])
    for seed in range(start_seed, end_seed + 1):
        for te_d in config['test_domains']:
            args.te_d = te_d
            set_seed(seed)
            args.seed = seed
            try:
                if torch.cuda.is_available():
                    logger.info("before experiment starts")
                    logger.info(str(torch.cuda.memory_summary()))
            except KeyError as ex:
                logger.error(str(ex))
            args.lr = float(args.lr)
            # <=' not supported between instances of 'float' and 'str
            exp = Exp(args=args, visitor=ExpProtocolAggWriter)
            # NOTE: if key "testing" is set in benchmark, then do not execute
            # experiment
            if not misc.get('testing', False):
                exp.execute()
            try:
                if torch.cuda.is_available():
                    logger.info("before torch memory clean up")
                    logger.info(str(torch.cuda.memory_summary()))
            except KeyError as ex:
                logger.error(str(ex))
            del exp
            torch.cuda.empty_cache()
            gc.collect()
            try:
                if torch.cuda.is_available():
                    logger.info("after torch memory clean up")
                    logger.info(str(torch.cuda.memory_summary()))
            except KeyError as ex:
                logger.error(str(ex))
