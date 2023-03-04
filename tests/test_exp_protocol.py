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
from domainlab.exp_protocol.aggregate_results import agg_results, agg_from_directory
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

    run_experiment(config, param_file, param_index, out_file, misc={'testing': True})
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


@pytest.fixture
def agg_input_files() -> List[str]:
    """Create test input files for the agg tests."""
    test_dir = "zoutput/test/rule_results"
    os.makedirs(test_dir, exist_ok=True)
    f_0 = test_dir + "/0.csv"
    f_1 = test_dir + "/1.csv"

    with open(f_0, 'w') as stream:
        stream.write(
            "param_index, task, algo, epos, te_d, seed, params, acc,"
            " precision, recall, specificity, f1, aurocy\n"
            "0, Task_diva, diva, 2, caltech, 1, \"{'gamma_y': 682408,"
            " 'gamma_d': 275835}\", 0.88461536, 0.852381,"
            " 0.80833334, 0.80833334, 0.82705104, 0.98333335\n"
        )

    with open(f_1, 'w') as stream:
        stream.write(
            "param_index, task, algo, epos, te_d, seed, params, acc,"
            " precision, recall, specificity, f1, aurocy\n"
            "1, TaskHduva, hduva, 2, caltech, 1, \"{'gamma_y': 70037,"
            " 'zy_dim': 48}\", 0.7307692, 0.557971,"
            " 0.5333333, 0.5333333, 0.5297158, 0.73333335"
        )

    # let the test run
    yield [f_0, f_1]
    # cleanup test files.
    shutil.rmtree(test_dir)


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
    return "param_index, task, algo, epos, te_d, seed, params, acc," \
           " precision, recall, specificity, f1, aurocy\n" \
           "0, Task_diva, diva, 2, caltech, 1, \"{'gamma_y': 682408," \
           " 'gamma_d': 275835}\", 0.88461536, 0.852381,"\
           " 0.80833334, 0.80833334, 0.82705104, 0.98333335\n"\
           "1, TaskHduva, hduva, 2, caltech, 1, \"{'gamma_y': 70037," \
           " 'zy_dim': 48}\", 0.7307692, 0.557971,"\
           " 0.5333333, 0.5333333, 0.5297158, 0.73333335"


def compare_file_content(filename: str, expected: str) -> bool:
    """Returns true if the given file contains the given string."""
    with open(filename, 'r') as stream:
        content = stream.readlines()
    return ''.join(content) == expected


def test_agg_results(agg_input_files, agg_output_file, agg_expected_output):
    """Testing the csv aggregation from a list of files."""
    agg_results(agg_input_files, agg_output_file)
    compare_file_content(agg_output_file, agg_expected_output)


def test_agg_from_directory(agg_input_files, agg_output_file, agg_expected_output):
    """Testing the csv aggregation from a full directory."""
    directory = os.path.dirname(agg_input_files[0])
    agg_from_directory(directory, agg_output_file)
    compare_file_content(agg_output_file, agg_expected_output)
