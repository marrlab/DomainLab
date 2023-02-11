import yaml

from domainlab.exp_protocol.run_experiment import run_experiment


def test_run_experiment():
    """Checks the run_experiment function on a minimal basis"""
    with open("examples/yaml/demo_benchmark.yaml", "r") as stream:
        config = yaml.safe_load(stream)

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
