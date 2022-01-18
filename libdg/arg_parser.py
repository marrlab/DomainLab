"""
Command line arguments
"""
import argparse
import warnings
from libdg.models.args_vae import add_args2parser_vae
from libdg.algos.compos.matchdg_args import add_args2parser_matchdg


def mk_parser_main():
    """
    Args for command line definition
    """
    parser = argparse.ArgumentParser(description='LibDG')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')

    parser.add_argument('--es', type=int, default=10,
                        help='early stop steps')

    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')

    parser.add_argument('--nocu', action='store_true', default=False,
                        help='disables CUDA')

    parser.add_argument('--gen', action='store_true', default=False,
                        help='save generated images')

    parser.add_argument('--keep_model', action='store_true', default=False,
                        help='do not delete model at the end of training')

    parser.add_argument('--epos', default=2, type=int,
                        help='maximum number of epochs')

    parser.add_argument('--epo_te', default=1, type=int,
                        help='test performance per {} epochs')

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--dmem', action='store_true', default=False)

    parser.add_argument('--out', type=str, default="zoutput",
                        help='absolute directory to store outputs')

    parser.add_argument('--dpath', type=str, default="zdpath",
                        help="path for dataset")

    parser.add_argument('--tpath', type=str, default=None,
                        help="path for custom task, should implement get_task function")

    parser.add_argument('--exptag', type=str, default="exptag",
                        help='tag as prefix of result aggregation file name \
                        e.g. git hash for reproducibility')

    parser.add_argument('--aggtag', type=str, default="aggtag",
                        help='tag in each line of result aggregation file \
                        e.g., to specify potential different configurations')

    parser.add_argument('--msel', type=str, default=None,
                        help='model selection: val, elbo, recon, the \
                        elbo and recon only make sense for vae models,\
                        will be ignored by other methods')

    parser.add_argument('--aname', metavar="an", type=str,
                        default='diva',
                        help='algorithm name')

    parser.add_argument('--acon', metavar="ac", type=str, default=None,
                        help='algorithm configuration name, (default None)')

    parser.add_argument('--task', metavar="ta", type=str,
                        help='task name')

    arg_group_task = parser.add_argument_group('task args')

    arg_group_task.add_argument('--bs', type=int, default=100,
                                help='loader batch size for mixed domains')

    arg_group_task.add_argument('--te_d', nargs='*', default=None,
                                help='test domain names separated by single space, \
                                will be parsed to be list of strings')

    arg_group_vae = parser.add_argument_group('vae')
    arg_group_vae = add_args2parser_vae(arg_group_vae)
    arg_group_matchdg = parser.add_argument_group('matchdg')
    arg_group_matchdg = add_args2parser_matchdg(arg_group_matchdg)
    return parser


def parse_cmd_args():
    """
    get args from command line
    """
    parser = mk_parser_main()
    args = parser.parse_args()
    if args.acon is None:
        print("\n\n")
        warnings.warn("no algorithm conf specified, going to use default")
        print("\n\n")
    return args
