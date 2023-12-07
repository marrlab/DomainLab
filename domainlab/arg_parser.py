"""
Command line arguments
"""
import argparse
import warnings

import yaml

from domainlab.algos.trainers.compos.matchdg_args import add_args2parser_matchdg
from domainlab.algos.trainers.args_dial import add_args2parser_dial
from domainlab.models.args_jigen import add_args2parser_jigen
from domainlab.models.args_vae import add_args2parser_vae
from domainlab.utils.logger import Logger


def mk_parser_main():
    """
    Args for command line definition
    """
    parser = argparse.ArgumentParser(description='DomainLab')

    parser.add_argument('-c', "--config", default=None,
                        help="load YAML configuration", dest="config_file",
                        type=argparse.FileType(mode='r'))

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')

    parser.add_argument('--gamma_reg', type=float, default=0.1,
                        help='weight of regularization loss')

    parser.add_argument('--es', type=int, default=1,
                        help='early stop steps')

    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')

    parser.add_argument('--nocu', action='store_true', default=False,
                        help='disables CUDA')

    parser.add_argument('--device', type=str, default=None,
                        help='device name default None')

    parser.add_argument('--gen', action='store_true', default=False,
                        help='save generated images')

    parser.add_argument('--keep_model', action='store_true', default=False,
                        help='do not delete model at the end of training')

    parser.add_argument('--epos', default=2, type=int,
                        help='maximum number of epochs')

    parser.add_argument('--epos_min', default=0, type=int,
                        help='maximum number of epochs')

    parser.add_argument('--epo_te', default=1, type=int,
                        help='test performance per {} epochs')
    
    parser.add_argument('-w', '--warmup', type=int, default=100,
                        help='number of epochs for hyper-parameter warm-up. \
                        Set to 0 to turn warmup off.')

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--dmem', action='store_true', default=False)
    parser.add_argument('--no_dump', action='store_true', default=False,
                        help='suppress saving the confusion matrix')

    parser.add_argument('--trainer', type=str, default=None,
                        help='specify which trainer to use')

    parser.add_argument('--out', type=str, default="zoutput",
                        help='absolute directory to store outputs')

    parser.add_argument('--dpath', type=str, default="zdpath",
                        help="path for storing downloaded dataset")

    parser.add_argument('--tpath', type=str, default=None,
                        help="path for custom task, should implement \
                        get_task function")

    parser.add_argument('--npath', type=str, default=None,
                        help="path of custom neural network for feature \
                        extraction")

    parser.add_argument('--npath_dom', type=str, default=None,
                        help="path of custom neural network for feature \
                        extraction")

    parser.add_argument('--npath_argna2val', action='append',
                        help="specify new arguments and their value \
                        e.g. '--npath_argna2val my_custom_arg_na \
                        --npath_argna2val xx/yy/zz.py', additional \
                        pairs can be appended")

    parser.add_argument('--nname_argna2val', action='append',
                        help="specify new arguments and their values \
                        e.g. '--nname_argna2val my_custom_network_arg_na \
                        --nname_argna2val alexnet', additional pairs \
                        can be appended")

    parser.add_argument('--nname', type=str, default=None,
                        help="name of custom neural network for feature \
                        extraction of classification")

    parser.add_argument('--nname_dom', type=str, default=None,
                        help="name of custom neural network for feature \
                        extraction of domain")

    parser.add_argument('--apath', type=str, default=None,
                        help="path for custom AlgorithmBuilder")

    parser.add_argument('--exptag', type=str, default="exptag",
                        help='tag as prefix of result aggregation file name \
                        e.g. git hash for reproducibility')

    parser.add_argument('--aggtag', type=str, default="aggtag",
                        help='tag in each line of result aggregation file \
                        e.g., to specify potential different configurations')

    parser.add_argument('--agg_partial_bm', type=str,
                        default=None, dest="bm_dir",
                        help="Aggregates and plots partial data of a snakemake \
                        benchmark. Requires the benchmark config file. \
                        Other arguments will be ignored.")

    parser.add_argument('--gen_plots', type=str,
                        default=None, dest="plot_data",
                        help="plots the data of a snakemake benchmark. "
                             "Requires the results.csv file"
                             "and an output file (specify by --outp_file,"
                             "default is zoutput/benchmarks/shell_benchmark). "
                             "Other arguments will be ignored.")

    parser.add_argument('--outp_dir', type=str,
                        default='zoutput/benchmarks/shell_benchmark', dest="outp_dir",
                        help="outpus file for the plots when creating them"
                             "using --gen_plots. "
                             "Default is zoutput/benchmarks/shell_benchmark")

    parser.add_argument('--param_idx', type=bool,
                        default=True, dest="param_idx",
                        help="True: parameter index is used in the "
                             "pots generated with --gen_plots."
                             "False: parameter name is used."
                             "Default is True.")

    parser.add_argument('--msel', choices=['val', 'loss_tr'], default="val",
                        help='model selection for early stop: val, loss_tr, recon, the \
                        elbo and recon only make sense for vae models,\
                        will be ignored by other methods')

    parser.add_argument('--aname', metavar="an", type=str,
                        default=None,
                        help='algorithm name')

    parser.add_argument('--acon', metavar="ac", type=str, default=None,
                        help='algorithm configuration name, (default None)')

    parser.add_argument('--task', metavar="ta", type=str,
                        help='task name')

    arg_group_task = parser.add_argument_group('task args')

    arg_group_task.add_argument('--bs', type=int, default=100,
                                help='loader batch size for mixed domains')

    arg_group_task.add_argument('--split', type=float, default=0,
                                help='proportion of training, a value between \
                                0 and 1, 0 means no train-validation split')

    arg_group_task.add_argument('--te_d', nargs='*', default=None,
                                help='test domain names separated by single space, \
                                will be parsed to be list of strings')

    arg_group_task.add_argument('--tr_d', nargs='*', default=None,
                                help='training domain names separated by \
                                single space, will be parsed to be list of \
                                strings; if not provided then all available \
                                domains that are not assigned to \
                                the test set will be used as training domains')

    arg_group_task.add_argument('--san_check', action='store_true', default=False,
                                help='save images from the dataset as a sanity check')

    arg_group_task.add_argument('--san_num', type=int, default=8,
                                help='number of images to be dumped for the sanity check')

    arg_group_task.add_argument('--loglevel', type=str, default='DEBUG',
                                help='sets the loglevel of the logger')

    # args for variational auto encoder
    arg_group_vae = parser.add_argument_group('vae')
    arg_group_vae = add_args2parser_vae(arg_group_vae)
    arg_group_matchdg = parser.add_argument_group('matchdg')
    arg_group_matchdg = add_args2parser_matchdg(arg_group_matchdg)
    arg_group_jigen = parser.add_argument_group('jigen')
    arg_group_jigen = add_args2parser_jigen(arg_group_jigen)
    args_group_dial = parser.add_argument_group('dial')
    args_group_dial = add_args2parser_dial(args_group_dial)
    return parser


def apply_dict_to_args(args, data: dict, extend=False):
    """
    Tries to apply the data to the args dict of DomainLab.
    Unknown keys are silently ignored as long as
    extend is not set.
    """
    arg_dict = args.__dict__
    for key, value in data.items():
        if (key in arg_dict) or extend:
            if isinstance(value, list):
                cur_val = arg_dict.get(key, None)
                if not isinstance(cur_val, list):
                    if cur_val is not None:
                        raise RuntimeError(f"input dictionary value is list, \
                                           however, in DomainLab args, we have {cur_val}, \
                                           going to overrite to list")
                    arg_dict[key] = []  # if args_dict[key] is None, cast it into a list
                    # domainlab will take care of it if this argument can not be a list
                arg_dict[key].extend(value)  # args_dict[key] is already a list
                # keep existing values for the list arg_dct[key]
            else:
                # over-write existing value
                arg_dict[key] = value
        else:
            raise ValueError("Unsupported key: ", key)


def parse_cmd_args():
    """
    get args from command line
    """
    parser = mk_parser_main()
    args = parser.parse_args()
    logger = Logger.get_logger(logger_name='main_out_logger', loglevel=args.loglevel)
    if args.config_file:
        data = yaml.safe_load(args.config_file)
        delattr(args, 'config_file')
        apply_dict_to_args(args, data)

    if args.acon is None and args.bm_dir is None:
        logger.warn("\n\n")
        logger.warn("no algorithm conf specified, going to use default")
        logger.warn("\n\n")
        warnings.warn("no algorithm conf specified, going to use default")

    return args
