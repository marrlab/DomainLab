from domainlab.arg_parser import parse_cmd_args
from domainlab.cli import domainlab_cli
from domainlab.exp.exp_cuda_seed import set_seed  # reproducibility
from domainlab.exp.exp_main import Exp
from domainlab.exp_protocol import aggregate_results
from domainlab.utils.generate_benchmark_plots import gen_benchmark_plots

if __name__ == "__main__":
    domainlab_cli()
