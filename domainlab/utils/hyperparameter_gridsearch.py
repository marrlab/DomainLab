'''
gridsearch for the hyperparameter space

def add_next_param_from_list is an recursive function to make cartesian product along all the scalar hyper-parameters, this resursive function is used
in def grid_task

'''
import os

import numpy as np
import pandas as pd
import domainlab.utils.hyperparameter_sampling as sampling
from domainlab.utils.logger import Logger


def round_to_discreate_grid_uniform(grid, param_config):
    '''
    round the values of the grid to the grid spacing specified in the config
    for uniform and loguniform grids
    '''
    if param_config['step'] == 0:
        return grid
    mini = param_config['min']
    maxi = param_config['max']

    if maxi - mini < param_config['step']:
        raise RuntimeError('distance between max and min to small for defined step size')

    discreate_gird = np.arange(mini, maxi, step=param_config['step'])
    for num, elem in enumerate(list(grid)):
        grid[num] = discreate_gird[(np.abs(discreate_gird - elem)).argmin()]
    return np.unique(grid)

def round_to_discreate_grid_normal(grid, param_config):
    '''
    round the values of the grid to the grid spacing specified in the config
    for normal and lognormal grids
    '''
    if param_config['step'] == 0:
        return grid
    # for normal and lognormal no min and max is provided
    # in this case the grid is constructed around the mean
    neg_steps = np.ceil((param_config['mean'] - np.min(grid)) /
                        param_config['step'])
    pos_steps = np.ceil((np.max(grid) - param_config['mean']) /
                        param_config['step'])
    mini = param_config['mean'] - param_config['step'] * neg_steps
    maxi = param_config['mean'] + param_config['step'] * pos_steps

    discreate_gird = np.arange(mini, maxi, step=param_config['step'])
    for num, elem in enumerate(list(grid)):
        grid[num] = discreate_gird[(np.abs(discreate_gird - elem)).argmin()]
    return np.unique(grid)

def uniform_grid(param_config):
    '''
    get a uniform distributed grid given the specifications in the param_config
    param_config: config which needs to contain 'num', 'max', 'min', 'step'
    '''
    num = param_config['num']
    maxi = float(param_config['max'])
    mini = float(param_config['min'])
    step = (maxi - mini) / num
    # linspace does exclude the end of the interval and include the beginning
    grid = np.linspace(mini + step / 2, maxi + step / 2, num)
    if 'step' in param_config.keys():
        return round_to_discreate_grid_uniform(grid, param_config)
    return grid

def loguniform_grid(param_config):
    '''
    get a loguniform distributed grid given the specifications in the param_config
    param_config: config which needs to contain 'num', 'max', 'min'
    '''
    num = param_config['num']
    maxi = np.log10(float(param_config['max']))
    mini = np.log10(float(param_config['min']))
    step = (maxi - mini) / num
    # linspace does exclude the end of the interval and include the beginning
    grid = 10 ** np.linspace(mini + step / 2, maxi + step / 2, num)
    if 'step' in param_config.keys():
        return round_to_discreate_grid_uniform(grid, param_config)
    return grid

def normal_grid(param_config, lognormal=False):
    '''
    get a normal distributed grid given the specifications in the param_config
    param_config: config which needs to contain 'num', 'mean', 'std'
    '''
    if param_config['num'] == 1:
        return np.array([param_config['mean']])
    # Box–Muller transform to get from a uniform distribution to a normal distribution
    num = int(np.floor(param_config['num'] / 2))
    step = 2 / (param_config['num'] + 1)
    # for a even number of samples
    if param_config['num'] % 2 == 0:
        param_grid = np.arange(step, 1, step=step)[:num]
        stnormal_grid = np.sqrt(-2 * np.log(param_grid))
        stnormal_grid = np.append(stnormal_grid, -stnormal_grid)
        stnormal_grid = stnormal_grid / np.std(stnormal_grid)
        stnormal_grid = param_config['std'] * stnormal_grid + param_config['mean']
    # for a odd number of samples
    else:
        param_grid = np.arange(step, 1, step=step)[:num]
        stnormal_grid = np.sqrt(-2 * np.log(param_grid))
        stnormal_grid = np.append(stnormal_grid, -stnormal_grid)
        stnormal_grid = np.append(stnormal_grid, 0)
        stnormal_grid = stnormal_grid / np.std(stnormal_grid)
        stnormal_grid = param_config['std'] * stnormal_grid + param_config['mean']

    if 'step' in param_config.keys() and lognormal is False:
        return round_to_discreate_grid_normal(stnormal_grid, param_config)
    return stnormal_grid

def lognormal_grid(param_config):
    '''
    get a normal distributed grid given the specifications in the param_config
    param_config: config which needs to contain 'num', 'mean', 'std'
    '''
    grid = 10 ** normal_grid(param_config, lognormal=True)
    if 'step' in param_config.keys():
        return round_to_discreate_grid_normal(grid, param_config)
    return grid

def add_next_param_from_list(param_grid: dict, grid: dict,
                             grid_df: pd.DataFrame):
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
            add_next_param_from_list(param_grid_new, grid_new, grid_df)
    else:
        # add sample to grid_df
        grid_df.loc[len(grid_df.index)] = [grid]

def add_references_and_check_constraints(grid_df_prior, grid_df, referenced_params,
                                         config, task_name):
    '''
    in the last step all parameters which are referenced need to be add to the
    grid. All gridpoints not satisfying the constraints are removed afterwards.
    '''
    for dictio in grid_df_prior['params']:
        for key, val in dictio.items():
            exec(f"{key} = val")
        # add referenced params
        for rev_param, val in referenced_params.items():
            val = eval(val)
            dictio.update({rev_param: val})
            exec(f"{rev_param} = val")
        # check constraints
        if config['hyperparameters'].get('constraints', None) is not None:
            accepted = True
            for constr in config['hyperparameters'].get('constraints', None):
                if not eval(constr):
                    accepted = False
            if accepted:
                grid_df.loc[len(grid_df.index)] = [task_name, config['aname'], dictio]
        else:
            grid_df.loc[len(grid_df.index)] = [task_name, config['aname'], dictio]

def sample_grid(param_config):
    '''
    given the parameter config, this function samples all parameters which are distributed
    according the the categorical, uniform, loguniform, normal or lognormal distribution.
    '''
    # sample cathegorical parameter
    if param_config['distribution'] == 'categorical':
        param_grid = sampling.CategoricalHyperparameter('', param_config).allowed_values
    # sample uniform parameter
    elif param_config['distribution'] == 'uniform':
        param_grid = uniform_grid(param_config)
    # sample loguniform parameter
    elif param_config['distribution'] == 'loguniform':
        param_grid = loguniform_grid(param_config)
    # sample normal parameter
    elif param_config['distribution'] == 'normal':
        param_grid = normal_grid(param_config)
    # sample lognormal parameter
    elif param_config['distribution'] == 'lognormal':
        param_grid = lognormal_grid(param_config)
    else:
        raise RuntimeError(f'distribution \"{param_config["distribution"]}\" not '
                           f'implemented use a distribution from '
                           f'[categorical, uniform, loguniform, normal, lognormal]')
    return param_grid

def grid_task(grid_df: pd.DataFrame, task_name: str, config: dict):
    """create grid for one task and add it to the dataframe"""
    if 'hyperparameters' in config.keys():
        param_grids = {}
        referenced_params = {}
        for param_name in config['hyperparameters'].keys():
            param_config = config['hyperparameters'][param_name]
            if not param_name == 'constraints':
                if not 'num' in param_config.keys() \
                        and not 'reference' in param_config.keys() \
                        and not param_config['distribution'] == 'categorical':
                    raise RuntimeError(f"the number of parameters in the grid direction "
                                       f"of {param_name} needs to be specified")

            # constraints are not parameters
            if not param_name == 'constraints':
                # remember all parameters which are reverenced
                if 'reference' in param_config.keys():
                    referenced_params.update({param_name: param_config['reference']})
                # sample other parameter
                elif param_name != 'constraints':
                    param_grids.update({param_name: sample_grid(param_config)})

        # create the grid from the individual parameter grids
        # constraints are not respected in this step
        grid_df_prior = pd.DataFrame(columns=['params'])
        add_next_param_from_list(param_grids, {}, grid_df_prior)

        # add referenced params and check constraints
        add_references_and_check_constraints(grid_df_prior, grid_df, referenced_params,
                                             config, task_name)
        if grid_df[grid_df['algo'] == config['aname']].shape[0] == 0:
            raise RuntimeError('No valid value found for this grid spacing, refine grid')
    else:
        # add single line if no varying hyperparameters are specified.
        grid_df.loc[len(grid_df.index)] = [task_name, config['aname'], {}]


def sample_gridsearch(config: dict,
                      dest: str = None,
                      sampling_seed: int = None) -> pd.DataFrame:
    """
    create the hyperparameters grid according to the given
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

    logger = Logger.get_logger()
    samples = pd.DataFrame(columns=['task', 'algo', 'params'])
    for key, val in config.items():
        if sampling.is_task(val):
            grid_task(samples, key, val)
            logger.info(f'number of gridpoints for {key} : '
                        f'{samples[samples["algo"] == val["aname"]].shape[0]}')

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    logger.info(f'number of total sampled gridpoints: {samples.shape[0]}')
    samples.to_csv(dest)
    return samples
