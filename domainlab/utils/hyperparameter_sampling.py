from typing import Union

import numpy as np
import pandas as pd
import yaml


class Hyperparameter:
    """
    Represents a hyperparameter

    p1: min or mean
    p2: max or scale
    """
    # def __init__(self, name: str, p1: float, p2: float, distribution: str, step: float):
    def __init__(self, name: str, config: dict):
        self.name = name
        self.step = config.get('step', 0)
        try:
            self.distribution = config['distribution']
            if self.distribution == 'uniform' or self.distribution == 'loguniform':
                self.p1 = config['min']
                self.p2 = config['max']
            elif self.distribution == 'normal' or self.distribution == 'lognormal':
                self.p1 = config['mean']
                self.p2 = config['std']
            else:
                raise RuntimeError(f"Unsupported distribution type: {self.distribution}.")
        except KeyError:
            raise RuntimeError(f"Missing required key for parameter {name}.")

        self.p1 = float(self.p1)
        self.p2 = float(self.p2)
        self.val = 0

    def _ensure_step(self):
        """Make sure that the hyperparameter sticks to the discrete grid"""
        if self.step == 0:
            return   # continous parameter

        # round to next discrete value.
        off = (self.val - self.p1) % self.step
        if off < self.step / 2:
            self.val -= off
        else:
            self.val += self.step - off

    def sample(self):
        """Sample this parameter, respecting properties"""
        if self.distribution == 'uniform':
            self.val = np.random.uniform(self.p1, self.p2)
        elif self.distribution == 'loguniform':
            self.val = 10 ** np.random.uniform(np.log10(self.p1), np.log10(self.p2))
        elif self.distribution == 'normal':
            self.val = np.random.normal(self.p1, self.p2)
        elif self.distribution == 'lognormal':
            self.val = 10 ** np.random.normal(self.p1, self.p2)
        else:
            raise RuntimeError(f"Unsupported distribution type: {self.distribution}.")
        self._ensure_step()


def check_constraints(params: list[Hyperparameter], constraints: Union[list[str], None]) -> bool:
    """Check if the constraints are fulfilled."""
    if constraints is None:
        return True     # shortcut

    # set each param as a local variable
    for p in params:
        exec(f'{p.name} = {p.val}')
    # check all constraints
    for c in constraints:
        try:
            b = eval(c)
        except SyntaxError:
            raise SyntaxError(f"Invalid syntax in yaml config: {c}")
        if not b:
            return False

    return True


def sample_parameters(params: list[Hyperparameter], constraints: Union[list[str],None]) -> dict:
    """
    Tries to sample from the hyperparameter list.

    Errors if in 10_0000 attempts no sample complying with the
    constraints is found.
    """
    for i in range(10_000):
        for p in params:
            p.sample()
        if check_constraints(params, constraints):
            samples = {}
            for p in params:
                samples[p.name] = p.val
            return samples

    raise RuntimeError("Could not find an acceptable sample in 10,000 runs."
                       "Are the bounds and constraints reasonable?")


def sample_task(num_samples: int, sample_df: pd.DataFrame, task_name: str, config: dict):
    """Sample one task and add it to the dataframe"""
    algo = config['aname']
    if 'hyperparameters' in config.keys():
        params = []
        for key, val in config['hyperparameters'].items():
            if key == 'constraints':
                continue
            params += [Hyperparameter(key, val)]

        constraints = config['hyperparameters'].get('constraints', None)
        for i in range(num_samples):
            sample = sample_parameters(params, constraints)
            sample_df.loc[len(sample_df.index)] = [task_name, algo, sample]


def sample_hyperparameters(src: str, dest: str) -> pd.DataFrame:
    """
    Samples the hyperparameters according to the given
    yaml src file. The samples are saved to dest as csv

    Note: Parts of the yaml content are executed. Thus use this
    only with trusted config files.
    """
    with open(src, "r") as stream:
        config = yaml.safe_load(stream)

    num_samples = config['num_param_samples']
    samples = pd.DataFrame(columns=['task', 'algo', 'params'])
    for key, val in config.items():
        if isinstance(val, dict) and 'aname' in val.keys():
            sample_task(num_samples, samples, key, val)

    samples.to_csv(dest)
    return samples
