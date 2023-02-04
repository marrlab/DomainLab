import yaml

from domainlab.exp_protocol.run_experiment import run_experiment


def test_run_experiment():
    with open("examples/yaml/demo_benchmark.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    param_file = "data/ztest_files/test_parameter_samples.csv"
    param_index = 0
    out_file = "zoutput/benchmarks/demo_benchmark/rule_results/0.csv"
    test_domains = ['caltech']
    # misc = {'te_d': 'caltech'}
    misc = {}

    run_experiment(config, param_file, param_index, out_file, test_domains, misc)
