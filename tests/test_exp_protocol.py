"""
Tests run_experiment.py
"""
import os
import shutil
from typing import List

import pytest
import torch
import yaml

from domainlab.arg_parser import mk_parser_main
from domainlab.exp_protocol.aggregate_results import agg_results, agg_main
from domainlab.exp_protocol.run_experiment import run_experiment, apply_dict_to_args

def test_run_experiment():
    utils_run_experiment("examples/benchmark/demo_benchmark.yaml", list_test_domains=['caltech'])
    utils_run_experiment("examples/benchmark/demo_benchmark_mnist4test.yaml", ['0'], no_run=False)

def utils_run_experiment(yaml_name, list_test_domains, no_run=True):
    """Checks the run_experiment function on a minimal basis"""
    with open(yaml_name, "r", encoding="utf8") as stream:
        config = yaml.safe_load(stream)
    if torch.cuda.is_available():
        torch.cuda.init()
    config['epos'] = 1
    config['startseed'] = 1
    config['endseed'] = 1
    config['test_domains'] = list_test_domains
    param_file = "data/ztest_files/test_parameter_samples.csv"
    param_index = 0
    out_file = "zoutput/benchmarks/demo_benchmark/rule_results/0.csv"

    # setting misc={'testing': True} will disable experiment being executed
    run_experiment(config, param_file, param_index, out_file, misc={'testing': True})
    # setting test_domain equals zero will also not execute the experiment
    if no_run:
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


def create_agg_input_files() -> List[str]:
    """Creates test input files."""
    test_dir = "zoutput/test/rule_results"
    os.makedirs(test_dir, exist_ok=True)
    f_0 = test_dir + "/0.csv"
    f_1 = test_dir + "/1.csv"

    with open(f_0, 'w') as stream:
        stream.write(
            "param_index, method, algo, epos, te_d, seed, params, acc,"
            " precision, recall, specificity, f1, aurocy\n"
            "0, diva, diva, 2, caltech, 1, \"{'gamma_y': 682408,"
            " 'gamma_d': 275835}\", 0.88461536, 0.852381,"
            " 0.80833334, 0.80833334, 0.82705104, 0.98333335\n"
        )

    with open(f_1, 'w') as stream:
        stream.write(
            "param_index, method, algo, epos, te_d, seed, params, acc,"
            " precision, recall, specificity, f1, aurocy\n"
            "1, hduva, hduva, 2, caltech, 1, \"{'gamma_y': 70037,"
            " 'zy_dim': 48}\", 0.7307692, 0.557971,"
            " 0.5333333, 0.5333333, 0.5297158, 0.73333335"
        )
    return [f_0, f_1]


def cleanup_agg_test_files():
    """Delete temporal test files."""
    shutil.rmtree("zoutput/test", ignore_errors=True)


@pytest.fixture
def agg_input_files() -> List[str]:
    """Create test input files for the agg tests."""
    # let the test run
    yield create_agg_input_files()
    # cleanup test files.
    cleanup_agg_test_files()


@pytest.fixture
def agg_output_file() -> str:
    """Test output file for the agg tests."""
    # let the test run
    yield "zoutput/test/results.csv"
    # cleanup test files.
    os.remove("zoutput/test/results.csv")


@pytest.fixture
def agg_expected_output() -> str:
    """Expected result file content for the agg tests."""
    return "param_index, method, algo, epos, te_d, seed, params, acc," \
           " precision, recall, specificity, f1, aurocy\n" \
           "0, diva, diva, 2, caltech, 1, \"{'gamma_y': 682408," \
           " 'gamma_d': 275835}\", 0.88461536, 0.852381,"\
           " 0.80833334, 0.80833334, 0.82705104, 0.98333335\n"\
           "1, hduva, hduva, 2, caltech, 1, \"{'gamma_y': 70037," \
           " 'zy_dim': 48}\", 0.7307692, 0.557971,"\
           " 0.5333333, 0.5333333, 0.5297158, 0.73333335"


@pytest.fixture
def bm_config():
    """Test benchmark config."""
    create_agg_input_files()
    # let the test run
    yield "zoutput/test"
    # cleanup test files.
    cleanup_agg_test_files()


def compare_file_content(filename: str, expected: str) -> bool:
    """Returns true if the given file contains the given string."""
    with open(filename, 'r') as stream:
        content = stream.readlines()
    return ''.join(content) == expected


def test_agg_results(agg_input_files, agg_output_file, agg_expected_output):
    """Testing the csv aggregation from a list of files."""
    agg_results(agg_input_files, agg_output_file)
    compare_file_content(agg_output_file, agg_expected_output)


def test_agg_main(bm_config, agg_output_file, agg_expected_output):
    """Testing the csv aggregation from a full directory."""
    agg_main(bm_config, skip_plotting=True)
    compare_file_content(agg_output_file, agg_expected_output)
