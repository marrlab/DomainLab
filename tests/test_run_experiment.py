"""
Tests run_experiment.py
"""
import torch
import yaml

from domainlab.arg_parser import mk_parser_main
from domainlab.exp_protocol.run_experiment import run_experiment, apply_dict_to_args


def test_run_experiment():
    """Checks the run_experiment function on a minimal basis"""
    with open("examples/yaml/demo_benchmark.yaml", "r", encoding="utf8") as stream:
        config = yaml.safe_load(stream)
    if torch.cuda.is_available():
        torch.cuda.init()
    config['epos'] = 1
    config['startseed'] = 1
    config['endseed'] = 1
    config['test_domains'] = ['caltech']
    param_file = "data/ztest_files/test_parameter_samples.csv"
    param_index = 0
    out_file = "zoutput/benchmarks/demo_benchmark/rule_results/0.csv"

    run_experiment(config, param_file, param_index, out_file, {'testing': True})
    config['test_domains'] = []
    run_experiment(config, param_file, param_index, out_file)


def test_apply_dict_to_args():
    """Testing apply_dict_to_args"""
    parser = mk_parser_main()
    args = parser.parse_args(args=[])
    data = {'a': 1, 'b': [1, 2], 'aname': 'diva'}
    apply_dict_to_args(args, data, extend=True)
    assert args.a == 1
    assert args.aname == 'diva'
