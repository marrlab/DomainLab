import os

import numpy as np
import pandas as pd
import domainlab.utils.hyperparameter_sampling as sampling

def round_to_discreate_grid_uniform(grid, param_config):
    if param_config['step'] == 0:
        return grid
    else:
        min = param_config['min']
        max = param_config['max']

        discreate_gird = np.arange(min, max, step=param_config['step'])
        for num, elem in enumerate(list(grid)):
            grid[num] = discreate_gird[(np.abs(discreate_gird - elem)).argmin()]
        return np.unique(grid)

def round_to_discreate_grid_normal(grid, param_config):
    if param_config['step'] == 0:
        return grid
    else:
        neq_steps = np.ceil((param_config['mean'] - np.min(grid)) /
                                 param_config['step'])
        pos_steps = np.ceil((np.max(grid) - param_config['mean']) /
                                 param_config['step'])
        min = param_config['mean'] - param_config['step'] * neq_steps
        max = param_config['mean'] + param_config['step'] * pos_steps

        discreate_gird = np.arange(min, max, step=param_config['step'])
        for num, elem in enumerate(list(grid)):
            grid[num] = discreate_gird[(np.abs(discreate_gird - elem)).argmin()]
        return np.unique(grid)


def uniform_grid(param_config):
    '''
    get a uniform distributed grid given the specifications in the param_config
    param_config: config which needs to contain 'num', 'max', 'min', 'step'
    '''
    num = param_config['num']
    max = float(param_config['max'])
    min = float(param_config['min'])
    step = (max - min) / num
    # linspace does exclude the end of the interval and include the beginning
    grid = np.linspace(min + step / 2, max + step / 2, num)
    if 'step' in param_config.keys():
        return round_to_discreate_grid_uniform(grid, param_config)
    else:
        return grid

def loguniform_grid(param_config):
    '''
    get a loguniform distributed grid given the specifications in the param_config
    param_config: config which needs to contain 'num', 'max', 'min'
    '''
    num = param_config['num']
    max = np.log10(float(param_config['max']))
    min = np.log10(float(param_config['min']))
    step = (max - min) / num
    # linspace does exclude the end of the interval and include the beginning
    grid = 10 ** np.linspace(min + step / 2, max + step / 2, num)
    if 'step' in param_config.keys():
        return round_to_discreate_grid_uniform(grid, param_config)
    else:
        return grid

def normal_grid(param_config, lognormal=False):
    '''
    get a normal distributed grid given the specifications in the param_config
    param_config: config which needs to contain 'num', 'mean', 'std'
    '''
    # use Box–Muller transform to get from a uniform distribution to a normal distribution
    num = int(np.floor(param_config['num'] / 2))
    step = 2 / (param_config['num'] + 1)
    # for a even number of samples
    if param_config['num'] % 2 == 0:
        param_grid = np.arange(step, 1, step=step)[:num]
        normal_grid = np.sqrt(-2 * np.log(param_grid))
        normal_grid = np.append(normal_grid, -normal_grid)
        normal_grid = normal_grid / np.std(normal_grid)
        normal_grid = param_config['std'] * normal_grid + param_config['mean']
    # for a odd number of samples
    else:
        param_grid = np.arange(step, 1, step=step)[:num]
        normal_grid = np.sqrt(-2 * np.log(param_grid))
        normal_grid = np.append(normal_grid, -normal_grid)
        normal_grid = np.append(normal_grid, 0)
        normal_grid = normal_grid / np.std(normal_grid)
        normal_grid = param_config['std'] * normal_grid + param_config['mean']

    if 'step' in param_config.keys() and lognormal == False:
        return round_to_discreate_grid_normal(normal_grid, param_config)
    else:
        return normal_grid

def lognormal_grid(param_config):
    '''
    get a normal distributed grid given the specifications in the param_config
    param_config: config which needs to contain 'num', 'mean', 'std'
    '''
    grid = 10 ** normal_grid(param_config, lognormal=True)
    if 'step' in param_config.keys():
        return round_to_discreate_grid_normal(grid, param_config)
    else:
        return grid

def add_next_param_from_list(param_grid: dict, grid: dict,
                             grid_df: pd.DataFrame,
                             task_name: str, algo: str):
    '''
    can be used in a recoursive fassion to add all combinations of the parameters in
    param_grid to grid_df
    param_grid: dictionary with all possible values for each parameter
                {'p1': [1, 2, 3], 'p2': [0, 5], ...}
    grid: a grid which will build itself in the recursion, start with grid = {}
            after one step grid = {p1: 1}
    grid_df: dataframe which will save the finished grids
    task_name: task name
    also: algo name
    '''
    if len(param_grid.keys()) != 0:
        # specify the parameter to be used
        param_name = list(param_grid.keys())[0]
        # for all values of this parameter perform
        for param in param_grid[param_name]:
            # add the parameter to the grid
            grid_new = dict(grid)
            grid_new.update({param_name: param})
            # remove the parameter from param_grid
            param_grid_new = dict(param_grid)
            param_grid_new.pop(param_name)
            # resume with the next parameter
            add_next_param_from_list(param_grid_new, grid_new, grid_df, task_name, algo)
    else:
        # add sample to grid_df
        grid_df.loc[len(grid_df.index)] = [task_name, algo, grid]

def grid_task(grid_df: pd.DataFrame, task_name: str, config: dict):
    """Sample one task and add it to the dataframe"""
    algo = config['aname']
    if 'hyperparameters' in config.keys():
        constraints = config['hyperparameters'].get('constraints', None)
        param_grids = {}
        referenced_params = {}
        for param_name in config['hyperparameters'].keys():
            param_config = config['hyperparameters'][param_name]
            if not 'hyperparameters' in config.keys():
                RuntimeError(f"the number of parameters in the grid direction "
                             f"of {param_name} needs to be specified")

            # constraints are not parameters
            if param_name == 'constraints':
                continue
            # remember all parameters which are reverenced
            elif 'reference' in param_config.keys():
                referenced_params.update({param_name: param_config['reference']})
            # sample cathegorical parameter
            elif param_config['distribution'] == 'categorical':
                param_grid = sampling.CategoricalHyperparameter(param_name, param_config).allowed_values
                param_grids.update({param_name: param_grid})
            # sample uniform parameter
            elif param_config['distribution'] == 'uniform':
                param_grid = uniform_grid(param_config)
                param_grids.update({param_name: param_grid})
            # sample loguniform parameter
            elif param_config['distribution'] == 'loguniform':
                param_grid = loguniform_grid(param_config)
                param_grids.update({param_name: param_grid})
            # sample normal parameter
            elif param_config['distribution'] == 'normal':
                param_grid = normal_grid(param_config)
                param_grids.update({param_name: param_grid})
            # sample lognormal parameter
            elif param_config['distribution'] == 'lognormal':
                param_grid = lognormal_grid(param_config)
                param_grids.update({param_name: param_grid})

        grid = {}
        grid_df_prior = pd.DataFrame(columns=grid_df.columns)
        add_next_param_from_list(param_grids, grid, grid_df_prior, task_name, algo)

        # add referenced params and check constraints
        for dict in grid_df_prior['params']:
            for key, val in dict.items():
                exec("%s = val" % key)
            for rev_param in referenced_params.keys():
                val = eval(referenced_params[rev_param])
                dict.update({rev_param: val})
                exec("%s = val" % rev_param)
            # check constraints
            if constraints is not None:
                accepted = True
                for constr in constraints:
                    if not eval(constr):
                        accepted = False
                if accepted:
                    grid_df.loc[len(grid_df.index)] = [task_name, algo, dict]
            else:
                grid_df.loc[len(grid_df.index)] = [task_name, algo, dict]
        if grid_df[grid_df['algo'] == algo].shape[0] == 0:
            RuntimeError('No valid value found for this grid spacing, refine your grid')
    else:
        # add single line if no varying hyperparameters are specified.
        grid_df.loc[len(grid_df.index)] = [task_name, algo, {}]


def sample_gridsearch(config: dict,
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

    samples = pd.DataFrame(columns=['task', 'algo', 'params'])
    for key, val in config.items():
        if sampling.is_task(val):
            grid_task(samples, key, val)

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    samples.to_csv(dest)
    return samples


