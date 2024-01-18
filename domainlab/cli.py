"""
command-line interface (CLI) for the domainlab package
"""

from domainlab.arg_parser import parse_cmd_args
from domainlab.exp.exp_cuda_seed import set_seed  # reproducibility
from domainlab.exp.exp_main import Exp
from domainlab.exp_protocol import aggregate_results
from domainlab.utils.generate_benchmark_plots import gen_benchmark_plots


def domainlab_cli():
    """
    Function used to run domainlab as a command line tool for the package installed with pip.
    """
    args = parse_cmd_args()
    if args.bm_dir:
        aggregate_results.agg_main(args.bm_dir)
    elif args.plot_data is not None:
        gen_benchmark_plots(
            args.plot_data, args.outp_dir, use_param_index=args.param_idx
        )
    else:
        set_seed(args.seed)
        exp = Exp(args=args)
        exp.execute()
