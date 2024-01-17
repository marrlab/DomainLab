"""
Tests the ExpProtocolAggWriter
"""
from domainlab.arg_parser import mk_parser_main
from domainlab.exp.exp_main import Exp
from domainlab.exp.exp_utils import ExpProtocolAggWriter
from domainlab.exp_protocol.run_experiment import apply_dict_to_args
from domainlab.utils.get_git_tag import get_git_tag


def test_git_tag():
    """
    test if git tag could be generated
    """
    get_git_tag()


def test_exp_protocol_agg_writer():
    """Test the csv writer for the benchmark"""
    parser = mk_parser_main()
    args = parser.parse_args(args=[])
    misc = {
        "model": "diva",
        "nname": "conv_bn_pool_2",
        "nname_dom": "conv_bn_pool_2",
        "task": "mnistcolor10",
        "te_d": 0,
        "result_file": "out_file",
        "params": "hyperparameters",
        "benchmark_task_name": "task",
        "param_index": 0,
    }
    apply_dict_to_args(args, misc, extend=True)

    exp = Exp(args=args, visitor=ExpProtocolAggWriter)
    exp.visitor.get_cols()
    exp.visitor.get_fpath()
