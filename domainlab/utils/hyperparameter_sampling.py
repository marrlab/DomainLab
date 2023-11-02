"""
Samples the hyperparameters according to a benchmark configuration file.

# Structure of this file:
- Class Hyperparameter
# Inherited Classes
# Functions to sample hyper-parameters and log into csv file
"""
import copy
import os
import json
from pydoc import locate
from typing import List
from ast import literal_eval   # literal_eval can safe evaluate python expression

import numpy as np
import pandas as pd

from domainlab.utils.logger import Logger
from domainlab.utils.get_git_tag import get_git_tag


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
        if 'datatype' not in config:
            raise RuntimeError("Please specifiy datatype for all categorical hyper-parameters!, e.g. datatype=str")
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
                logger = Logger.get_logger()
                logger.info(f"error in evaluating expression: {par.reference}")
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


def sample_parameters(init_params: List[Hyperparameter], constraints,
                      shared_config=None, shared_samples=None) -> dict:
    """
    Tries to sample from the hyperparameter list.

    Errors if in 10_0000 attempts no sample complying with the
    constraints is found.
    """
    for _ in range(10_000):
        params = copy.deepcopy(init_params)
        for par in params:
            par.sample()
        # add a random hyperparameter from the shared hyperparameter dataframe
        if shared_samples is not None:
            # sample one line from the pandas dataframe
            shared_samp = shared_samples.sample(1).iloc[0]['params']
            for key in shared_samp.keys():
                par = Hyperparameter(key)
                par.val = shared_samp[key]
                par.name = key
                params.append(par)
        # check constrained
        if check_constraints(params, constraints):
            samples = {}
            for par in params:
                samples[par.name] = par.val
            return samples

    # if there was no sample found fullfilling the constrained above,
    # this may be due to the shared hyperparameters.
    # If so, new samples are generated for the shared hyperparameters
    logger = Logger.get_logger()
    logger.warning("The constrainds coundn't be met with the shared Hyperparameters, "
                   "shared dataframe pool will be ignored for now.")
    for _ in range(10_000):
        params = copy.deepcopy(init_params)
        # add the shared hyperparameter as a sampled hyperparameter
        if shared_samples is not None:
            shared_samp = shared_samples.sample(1).iloc[0]['params']
            for key in shared_samp.keys():
                par = SampledHyperparameter(key, shared_config[key])
                par.sample()
                par.name = key
                params.append(par)
        for par in params:
            par.sample()
        # check constrained
        if check_constraints(params, constraints):
            samples = {}
            for par in params:
                samples[par.name] = par.val
            return samples

    raise RuntimeError("Could not find an acceptable sample in 10,000 runs."
                       "Are the bounds and constraints reasonable?")


def create_samples_from_shared_samples(shared_samples: pd.DataFrame,
                                       config: dict,
                                       task_name: str):
    '''
    add informations like task, algo and constrainds to the shared samples
    Parameters:
    shared_samples: pd Dataframe with columns ['task', 'algo', 'params']
    config: dataframe with yaml configuration of the current task
    task_name: name of the current task
    '''
    shared_samp = shared_samples.copy()
    shared_samp['algo'] = config['aname']
    shared_samp['task'] = task_name
    # respect the constraints if specified in the task
    if 'constraints' in config.keys():
        for idx in range(shared_samp.shape[0] - 1, -1, -1):
            name = list(shared_samp['params'].iloc[idx].keys())[0]
            value = shared_samp['params'].iloc[idx][name]
            par = Hyperparameter(name)
            par.val = value
            if not check_constraints([par], config['constraints']):
                shared_samp = shared_samp.drop(idx)
    return shared_samp

def sample_task_only_shared(num_samples, task_name, sample_df, config, shared_conf_samp):
    '''
    sample one task and add it to the dataframe for task descriptions which only
    contain shared hyperparameters
    '''
    shared_config, shared_samples = shared_conf_samp
    # copy the shared samples dataframe and add the corrct algo and taks names
    shared_samp = create_samples_from_shared_samples(shared_samples, config, task_name)

    # for the case that we expect more hyperparameter samples for the algorithm as provided
    # in the shared sampes we use the shared config to sample new hyperparameters to ensure
    # that we have distinct hyperparameters
    if num_samples - shared_samp.shape[0] > 0:
        s_config = shared_config.copy()
        s_dict = {}
        for keys in s_config.keys():
            if keys != 'num_shared_param_samples':
                s_dict[keys] = s_config[keys]
        if 'constraints' in config.keys():
            s_dict['constraints'] = config['constraints']
        s_config['aname'] = config['aname']
        s_config['hyperparameters'] = s_dict

        # sample new shared hyperparameters
        sample_df = sample_task(num_samples - shared_samp.shape[0],
                                task_name, (s_config, sample_df), (None, None))
        # add previously sampled shared hyperparameters
        sample_df = sample_df.append(shared_samp, ignore_index=True)
    # for the case that the number of shared samples is >= the expected number of
    # sampled hyperparameters we randomly choose rows in the sampled hyperparameters df
    else:
        shared_samp = shared_samp.sample(num_samples)
        sample_df = sample_df.append(shared_samp, ignore_index=True)

    return sample_df

def sample_task(num_samples: int,
                task_name: str,
                conf_samp: tuple,
                shared_conf_samp: tuple):
    """Sample one task and add it to the dataframe"""
    config, sample_df = conf_samp
    shared_config, shared_samples = shared_conf_samp
    if 'hyperparameters' in config.keys():
        # in benchmark configuration file, sub-section hyperparameters
        # means changing hyper-parameters
        params = []
        for key, val in config['hyperparameters'].items():
            if key in ('constraints', 'num_shared_param_samples'):
                continue
            params += [get_hyperparameter(key, val)]

        constraints = config['hyperparameters'].get('constraints', None)
        for _ in range(num_samples):
            sample = sample_parameters(params, constraints, shared_config, shared_samples)
            sample_df.loc[len(sample_df.index)] = [task_name, config['aname'], sample]
    elif 'shared' in config.keys():
        sample_df = sample_task_only_shared(num_samples, task_name, sample_df,
                                            config, (shared_config, shared_samples))
    else:
        # add single line if no varying hyperparameters are specified.
        sample_df.loc[len(sample_df.index)] = [task_name, config['aname'], {}]
    return sample_df


def is_dict_with_key(input_dict, key) -> bool:
    """Determines if the input argument is a dictionary and it has key"""
    return isinstance(input_dict, dict) and key in input_dict.keys()

def get_shared_samples(shared_samples_full: pd.DataFrame,
                       shared_config_full: dict,
                       task_config: dict):
    '''
    - creates a dataframe with columns [task, algo, params],
    task and algo are all for all rows, but params is filled with the
    shared parameters of shared_samples_full requested by task_config.

    - creates a shared config containing only information about the
    shared hyperparameters requested by the task_config
    '''
    shared_samples = shared_samples_full.copy(deep=True)
    shared_config = shared_config_full.copy()
    if 'shared' in task_config.keys():
        shared = task_config['shared']
    else:
        shared = []
    for line_num in range(shared_samples.shape[0]):
        hyper_p_dict = shared_samples.iloc[line_num]['params'].copy()
        key_list = copy.deepcopy(list(hyper_p_dict.keys()))
        for key_ in key_list:
            if key_ not in shared:
                del hyper_p_dict[key_]
        shared_samples.iloc[line_num]['params'] = hyper_p_dict
    for key_ in key_list:
        if not key_ == 'num_shared_param_samples':
            if key_ not in shared:
                del shared_config[key_]
    # remove all duplicates
    shared_samples = shared_samples.drop_duplicates(subset='params')
    return shared_samples, shared_config

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

    if sampling_seed is not None:
        np.random.seed(sampling_seed)

    num_samples = config['num_param_samples']
    samples = pd.DataFrame(columns=['task', 'algo', 'params'])
    if 'Shared params' in config.keys():
        shared_config_full = config['Shared params']
        shared_samples_full = pd.DataFrame(columns=['task', 'algo', 'params'])
        shared_val = {'aname': 'all', 'hyperparameters':  config['Shared params']}
        # fill up the dataframe shared samples
        shared_samples_full = sample_task(shared_config_full['num_shared_param_samples'],
                                          'all', (shared_val, shared_samples_full), (None, None))
    else:
        shared_samples_full = None
    for key, val in config.items():
        if is_dict_with_key(val, "aname"):
            if shared_samples_full is not None:
                shared_samples, shared_config = get_shared_samples(
                    shared_samples_full, shared_config_full, val)
            else:
                shared_config = None
                shared_samples = None

            samples = sample_task(num_samples, key, (val, samples),
                                  (shared_config, shared_samples))

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # create a txt file with the commit information
    with open(config["output_dir"] + os.sep + 'commit.txt', 'w', encoding="utf8") as file:
        file.writelines("use git log |grep \n")
        file.writelines("consider remove leading b in the line below \n")
        file.write(get_git_tag())
    with open(config["output_dir"] + os.sep + 'config.txt', 'w', encoding="utf8") as file:
        json.dump(config, file)

    samples.to_csv(dest)
    return samples
