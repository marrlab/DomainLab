"""
tests hyperparameter_sampling.py
"""
import pandas as pd
import pytest
import yaml

from domainlab.utils.hyperparameter_sampling import\
    sample_hyperparameters, Hyperparameter, sample_parameters


def test_hyperparameter_sampling():
    """Test sampling from yaml, including constraints"""
    with open("examples/yaml/demo_hyperparameter_sampling.yml", "r") as stream:
        config = yaml.safe_load(stream)

    samples = sample_hyperparameters(config)

    a1samples = samples[samples['algo'] == 'Algo1']
    for par in a1samples['params']:
        assert par['p1'] < par['p2']
        assert par['p3'] < par['p2']
        assert par['p2'] % 1 == pytest.approx(0)
        assert par['p4'] == par['p3']

    a2samples = samples[samples['algo'] == 'Algo2']
    for par in a2samples['params']:
        assert par['p1'] % 2 == pytest.approx(1)
        assert par['p2'] % 1 == pytest.approx(0)

    a3samples = samples[samples['algo'] == 'Algo3']
    assert len(a3samples) > 0


def test_hyperparameter_errors():
    """Test for errors on unknown distribution or missing keys"""
    with pytest.raises(RuntimeError, match='Unsupported distribution'):
        Hyperparameter('name', {'distribution': 'unknown'})

    with pytest.raises(RuntimeError, match='Missing required key'):
        Hyperparameter('name', {'distribution': 'uniform'})

    par = Hyperparameter('name', {'distribution': 'uniform', 'min': 0, 'max': 1})
    par.distribution = 'unknown'
    with pytest.raises(RuntimeError, match='Unsupported distribution'):
        par.sample()
    par.get_val()


def test_constraint_error():
    """Check error on invalid syntax in constraints"""
    par = Hyperparameter('name', {'distribution': 'uniform', 'min': 0, 'max': 1})
    constraints = ["hello world"]
    with pytest.raises(SyntaxError, match='Invalid syntax in yaml config'):
        sample_parameters([par], constraints)


def test_sample_parameters_abort():
    """Test for error on infeasible constraints"""
    p_1 = Hyperparameter('p1', {'distribution': 'uniform', 'min': 0, 'max': 1})
    p_2 = Hyperparameter('p2', {'distribution': 'uniform', 'min': 2, 'max': 3})
    constraints = ['p2 < p1']   # impossible due to the bounds
    with pytest.raises(RuntimeError, match='constraints reasonable'):
        sample_parameters([p_1, p_2], constraints)


def test_sampling_seed():
    """Tests if the same hyperparameters are sampled if sampling_seed is set"""
    with open("examples/yaml/demo_hyperparameter_sampling.yml", "r") as stream:
        config = yaml.safe_load(stream)

    config['sampling_seed'] = 1

    samples1 = sample_hyperparameters(config)
    samples2 = sample_hyperparameters(config)
    pd.testing.assert_frame_equal(samples1, samples2)
