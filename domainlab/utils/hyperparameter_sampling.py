"""
Samples the hyperparameters according to a benchmark configuration file.

# Structure of this file:
- Class Hyperparameter
# Inherited Classes
# Functions to sample hyper-parameters and log into csv file
"""
import os
from pydoc import locate
from typing import List
from ast import literal_eval   # literal_eval can safe evaluate python expression

import numpy as np
import pandas as pd


class Hyperparameter:
    """
    Represents a hyperparameter.
    The datatype of .val is int if step and p1 is integer valued,
    else float.

    p1: min or mean
    p2: max or scale
    reference: None or name of referenced hyperparameter
    """
    def __init__(self, name: str):
        self.name = name
        self.val = 0

    def _ensure_step(self):
        """Make sure that the hyperparameter sticks to the discrete grid"""
        raise NotImplementedError

    def sample(self):
        """Sample this parameter, respecting properties"""
        raise NotImplementedError

    def get_val(self):
        """Returns the current value of the hyperparameter"""
        return self.val

    def datatype(self):
        """
        Returns the datatype of this parameter.
        This does not apply for references.
        """
        raise NotImplementedError


class SampledHyperparameter(Hyperparameter):
    """
    A numeric hyperparameter that shall be sampled
    """
    def __init__(self, name: str, config: dict):
        super().__init__(name)
        self.step = config.get('step', 0)
        try:
            self.distribution = config['distribution']
            if self.distribution in {'uniform', 'loguniform'}:
                self.p_1 = config['min']
                self.p_2 = config['max']
            elif self.distribution in {'normal', 'lognormal'}:
                self.p_1 = config['mean']
                self.p_2 = config['std']
            else:
                raise RuntimeError(f"Unsupported distribution type: {self.distribution}.")
        except KeyError as ex:
            raise RuntimeError(f"Missing required key for parameter {name}.") from ex

        self.p_1 = float(self.p_1)
        self.p_2 = float(self.p_2)

    def _ensure_step(self):
        """Make sure that the hyperparameter sticks to the discrete grid"""
        if self.step == 0:
            return   # continous parameter

        # round to next discrete value.
        # p_1 is the lower bound of the hyper-parameter range, p_2 the upper bound
        off = (self.val - self.p_1) % self.step
        # $off$ is always smaller than step, depending on whether $off$ falls on the left half
        # or right half of [0, step], move the hyper-parameter to the boundary so that
        # updated hyper-parameter % step = 0
        if off < self.step / 2:
            self.val -= off
        else:
            self.val += self.step - off
        # ensure correct datatype
        if self.datatype() == int:
            self.val = self.datatype()(np.round(self.val))

    def sample(self):
        """Sample this parameter, respecting properties"""
        if self.distribution == 'uniform':
            self.val = np.random.uniform(self.p_1, self.p_2)
        elif self.distribution == 'loguniform':
            self.val = 10 ** np.random.uniform(np.log10(self.p_1), np.log10(self.p_2))
        elif self.distribution == 'normal':
            self.val = np.random.normal(self.p_1, self.p_2)
        elif self.distribution == 'lognormal':
            self.val = 10 ** np.random.normal(self.p_1, self.p_2)
        else:
            raise RuntimeError(f"Unsupported distribution type: {self.distribution}.")
        self._ensure_step()

    def datatype(self):
        return int if self.step % 1 == 0 and self.p_1 % 1 == 0 else float


class ReferenceHyperparameter(Hyperparameter):
    """
    Hyperparameter that references only a different one.
    Thus, this parameter is not sampled but set after sampling.
    """
    def __init__(self, name: str, config: dict):
        super().__init__(name)
        self.reference = config.get('reference', None)

    def _ensure_step(self):
        """Make sure that the hyperparameter sticks to the discrete grid"""
        # nothing to do for references
        return

    def sample(self):
        """Sample this parameter, respecting properties"""
        # nothing to do for references
        return

    def datatype(self):
        raise RuntimeError("Datatype unknown for ReferenceHyperparameter")


class CategoricalHyperparameter(Hyperparameter):
    """
    A sampled hyperparameter, which is constraint to fixed,
    user given values and datatype
    """
    def __init__(self, name: str, config: dict):
        super().__init__(name)
        self.allowed_values = config['values']
        self.type = locate(config['datatype'])
        self.allowed_values = [self.type(v) for v in self.allowed_values]

    def _ensure_step(self):
        """Make sure that the hyperparameter sticks to the discrete grid"""
        # nothing to do for categorical ones
        return

    def sample(self):
        """Sample this parameter, respecting properties"""
        # nothing to do for references
        idx = np.random.randint(0, len(self.allowed_values))
        self.val = self.allowed_values[idx]

    def datatype(self):
        return self.type


def get_hyperparameter(name: str, config: dict) -> Hyperparameter:
    """Factory function. Instantiates the correct Hyperparameter"""
    if 'reference' in config.keys():
        return ReferenceHyperparameter(name, config)
    dist = config.get('distribution', None)
    if dist == 'categorical':
        return CategoricalHyperparameter(name, config)

    return SampledHyperparameter(name, config)


def check_constraints(params: List[Hyperparameter], constraints) -> bool:
    """Check if the constraints are fulfilled."""
    # set each param as a local variable
    for par in params:
        locals().update({par.name: par.val})

    # set references
    for par in params:
        if isinstance(par, ReferenceHyperparameter):
            try:
                setattr(par, 'val', eval(par.reference))
                # NOTE: literal_eval will cause ValueError: malformed node or string
            except Exception as ex:
                print(f"error in evaluating expression: {par.reference}")
                raise ex
            locals().update({par.name: par.val})

    if constraints is None:
        return True     # shortcut

    # check all constraints
    for constr in constraints:
        try:
            const_res = eval(constr)
            # NOTE: literal_eval will cause ValueError: malformed node or string
        except SyntaxError as ex:
            raise SyntaxError(f"Invalid syntax in yaml config: {constr}") from ex
        if not const_res:
            return False

    return True


def sample_parameters(params: List[Hyperparameter], constraints) -> dict:
    """
    Tries to sample from the hyperparameter list.

    Errors if in 10_0000 attempts no sample complying with the
    constraints is found.
    """
    for _ in range(10_000):
        for par in params:
            par.sample()
        if check_constraints(params, constraints):
            samples = {}
            for par in params:
                samples[par.name] = par.val
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
            params += [get_hyperparameter(key, val)]

        constraints = config['hyperparameters'].get('constraints', None)
        for _ in range(num_samples):
            sample = sample_parameters(params, constraints)
            sample_df.loc[len(sample_df.index)] = [task_name, algo, sample]
    else:
        # add single line if no varying hyperparameters are specified.
        sample_df.loc[len(sample_df.index)] = [task_name, algo, {}]


def is_task(val) -> bool:
    """Determines if the value of this key is a task."""
    return isinstance(val, dict) and 'aname' in val.keys()


def sample_hyperparameters(config: dict,
                           dest: str = None,
                           sampling_seed: int = None) -> pd.DataFrame:
    """
    Samples the hyperparameters according to the given
    config, which should be the dictionary of the full
    benchmark config yaml.
    Result is saved to 'output_dir/hyperparameters.csv' of the
    config if not specified explicitly.

    Note: Parts of the yaml content are executed. Thus use this
    only with trusted config files.
    """
    if dest is None:
        dest = config['output_dir'] + os.sep + 'hyperparameters.csv'

    if not sampling_seed is None:
        np.random.seed(sampling_seed)

    num_samples = config['num_param_samples']
    samples = pd.DataFrame(columns=['task', 'algo', 'params'])
    for key, val in config.items():
        if is_task(val):
            sample_task(num_samples, samples, key, val)

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    samples.to_csv(dest)
    return samples
