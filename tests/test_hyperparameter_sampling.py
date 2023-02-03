import pytest

from domainlab.utils.hyperparameter_sampling import sample_hyperparameters, Hyperparameter, sample_parameters


def test_hyperparameter_sampling():
    """Test sampling from yaml, including constraints"""
    samples = sample_hyperparameters("./../configs/hyperparameter_test_config.yml", './../zoutput/test_params.csv')
    a1samples = samples[samples['algo'] == 'Algo1']
    for p in a1samples['params']:
        assert p['p1'] < p['p2']
        assert p['p3'] < p['p2']
        assert p['p2'] % 1 == pytest.approx(0)

    a2samples = samples[samples['algo'] == 'Algo2']
    for p in a2samples['params']:
        assert p['p1'] % 2 == pytest.approx(1)
        assert p['p2'] % 1 == pytest.approx(0)


def test_hyperparameter_errors():
    """Test for errors on unknown distribution or missing keys"""
    with pytest.raises(RuntimeError, match='Unsupported distribution'):
        Hyperparameter('name', {'distribution': 'unknown'})

    with pytest.raises(RuntimeError, match='Missing required key'):
        Hyperparameter('name', {'distribution': 'uniform'})

    p = Hyperparameter('name', {'distribution': 'uniform', 'min': 0, 'max': 1})
    p.distribution = 'unknown'
    with pytest.raises(RuntimeError, match='Unsupported distribution'):
        p.sample()


def test_constraint_error():
    """Check error on invalid syntax in constraints"""
    p = Hyperparameter('name', {'distribution': 'uniform', 'min': 0, 'max': 1})
    constraints = ["hello world"]
    with pytest.raises(SyntaxError, match='Invalid syntax in yaml config'):
        sample_parameters([p], constraints)


def test_sample_parameters_abort():
    """Test for error on infeasible constraints"""
    p1 = Hyperparameter('p1', {'distribution': 'uniform', 'min': 0, 'max': 1})
    p2 = Hyperparameter('p2', {'distribution': 'uniform', 'min': 2, 'max': 3})
    constraints = ['p2 < p1']   # impossible due to the bounds
    with pytest.raises(RuntimeError, match='constraints reasonable'):
        sample_parameters([p1, p2], constraints)
