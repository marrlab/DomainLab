'''
gridsearch for the hyperparameter space

def add_next_param_from_list is an recursive function to make cartesian product along all the scalar hyper-parameters, this resursive function is used
in def grid_task

'''
import copy
import os
import json
import warnings

import numpy as np
import pandas as pd
import domainlab.utils.hyperparameter_sampling as sampling
from domainlab.utils.get_git_tag import get_git_tag
from domainlab.utils.logger import Logger


def round_to_discreate_grid_uniform(grid, param_config):
    '''
    round the values of the grid to the grid spacing specified in the config
    for uniform and loguniform grids
    '''
    if float(param_config['step']) == 0:
        return grid
    mini = float(param_config['min'])
    maxi = float(param_config['max'])
    if maxi - mini < float(param_config['step']):
        raise RuntimeError('distance between max and min to small for defined step size')

    discreate_gird = np.arange(mini, maxi + float(param_config['step']),
                               step=float(param_config['step']))
    for num, elem in enumerate(list(grid)):
        # search for the closest allowed grid point to the scalar elem
        grid[num] = discreate_gird[(np.abs(discreate_gird - elem)).argmin()]
    grid_unique = np.unique(grid)
    grid_out = grid_unique
    return grid_out

def round_to_discreate_grid_normal(grid, param_config):
    '''
    round the values of the grid to the grid spacing specified in the config
    for normal and lognormal grids
    '''
    if float(param_config['step']) == 0:
        return grid
    # for normal and lognormal no min and max is provided
    # in this case the grid is constructed around the mean
    neg_steps = np.ceil((float(param_config['mean']) - np.min(grid)) /
                        float(param_config['step']))
    pos_steps = np.ceil((np.max(grid) - float(param_config['mean'])) /
                        float(param_config['step']))
    mini = float(param_config['mean']) - float(param_config['step']) * neg_steps
    maxi = float(param_config['mean']) + float(param_config['step']) * pos_steps

    discreate_gird = np.arange(mini, maxi, step=float(param_config['step']))
    for num, elem in enumerate(list(grid)):
        grid[num] = discreate_gird[(np.abs(discreate_gird - elem)).argmin()]
    return np.unique(grid)

def uniform_grid(param_config):
    '''
    get a uniform distributed grid given the specifications in the param_config
    param_config: config which needs to contain 'num', 'max', 'min', 'step'
    '''
    num = int(param_config['num'])
    maxi = float(param_config['max'])
    mini = float(param_config['min'])
    step = (maxi - mini) / num
    # linspace does include the end of the interval and include the beginning
    # we move away from mini and maxi to sample inside the open interval (mini, maxi)
    grid = np.linspace(mini + step / 2, maxi - step / 2, num)
    if 'step' in param_config.keys():
        return round_to_discreate_grid_uniform(
            grid, param_config)
    return grid

def loguniform_grid(param_config):
    '''
    get a loguniform distributed grid given the specifications in the param_config
    param_config: config which needs to contain 'num', 'max', 'min'
    '''
    num = int(param_config['num'])
    maxi = np.log10(float(param_config['max']))
    mini = np.log10(float(param_config['min']))
    step = (maxi - mini) / num
    # linspace does exclude the end of the interval and include the beginning
    grid = 10 ** np.linspace(mini + step / 2, maxi - step / 2, num)
    if 'step' in param_config.keys():
        return round_to_discreate_grid_uniform(grid, param_config)
    return grid

def normal_grid(param_config, lognormal=False):
    '''
    get a normal distributed grid given the specifications in the param_config
    param_config: config which needs to contain 'num', 'mean', 'std'
    '''
    if int(param_config['num']) == 1:
        return np.array([float(param_config['mean'])])
    # Box–Muller transform to get from a uniform distribution to a normal distribution
    num = int(np.floor(int(param_config['num']) / 2))
    step = 2 / (int(param_config['num']) + 1)
    # for a even number of samples
    if int(param_config['num']) % 2 == 0:
        param_grid = np.arange(step, 1, step=step)[:num]
        stnormal_grid = np.sqrt(-2 * np.log(param_grid))
        stnormal_grid = np.append(stnormal_grid, -stnormal_grid)
        stnormal_grid = stnormal_grid / np.std(stnormal_grid)
        stnormal_grid = float(param_config['std']) * stnormal_grid + \
                        float(param_config['mean'])
    # for a odd number of samples
    else:
        param_grid = np.arange(step, 1, step=step)[:num]
        stnormal_grid = np.sqrt(-2 * np.log(param_grid))
        stnormal_grid = np.append(stnormal_grid, -stnormal_grid)
        stnormal_grid = np.append(stnormal_grid, 0)
        stnormal_grid = stnormal_grid / np.std(stnormal_grid)
        stnormal_grid = float(param_config['std']) * stnormal_grid + \
                        float(param_config['mean'])

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
        if 'hyperparameters' in config.keys():
            constraints = config['hyperparameters'].get('constraints', None)
        else:
            constraints = config.get('constraints', None)
        if constraints is not None:
            accepted = True
            for constr in constraints:
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

    # ensure that the gird does have the correct datatype
    # (only check for int, othervise float is used)
    if 'datatype' in param_config.keys():
        if param_config['datatype'] == 'int':
            param_grid = np.array(param_grid)
            param_grid = param_grid.astype(int)
        # NOTE: converting int to float will cause error for VAE, avoid do
        # it here
        return param_grid

def build_param_grid_of_shared_params(shared_df):
    '''
    go back from the data frame format of the shared hyperparamters to a list format
    '''
    if shared_df is None:
        return None
    shared_grid = {}
    for key in shared_df['params'].iloc[0].keys():
        grid_points = []
        for i in shared_df['params'].keys():
            grid_points.append(shared_df['params'][i][key])
        shared_grid[key] = np.array(grid_points)
    return shared_grid

def rais_error_if_num_not_specified(param_name: str, param_config: dict):
    '''
    for each parameter a number of grid points needs to be specified
    This function raises an error if this is not the case
    param_name: parameter name under consideration
    param_config: config of this parameter
    '''
    if not param_name == 'constraints':
        if not 'num' in param_config.keys() \
                and not 'reference' in param_config.keys() \
                and not param_config['distribution'] == 'categorical':
            raise RuntimeError(f"the number of parameters in the grid direction "
                               f"of {param_name} needs to be specified")

def add_shared_params_to_param_grids(shared_df, dict_param_grids, config):
    '''
    use the parameters in the dataframe of shared parameters and add them
    to the dictionary of parameters for the current task
    only the shared parameters specified in the config are respected
    shared_df: Dataframe of shared hyperparameters
    dict_param_grids: dictionary of the parameter grids
    config: config for the current task
    '''
    dict_shared_grid = build_param_grid_of_shared_params(shared_df)
    if 'shared' in config.keys():
        list_names = config['shared']
        dict_shared_grid = {key: dict_shared_grid[key] for key in config['shared']}
        if dict_shared_grid is not None:
            for key in dict_shared_grid.keys():
                dict_param_grids[key] = dict_shared_grid[key]
    return dict_param_grids

def grid_task(grid_df: pd.DataFrame, task_name: str, config: dict, shared_df: pd.DataFrame):
    """create grid for one sampling task for a method and add it to the dataframe"""
    if 'hyperparameters' in config.keys():
        dict_param_grids = {}
        referenced_params = {}
        for param_name in config['hyperparameters'].keys():
            param_config = config['hyperparameters'][param_name]
            rais_error_if_num_not_specified(param_name, param_config)

            # constraints are not parameters
            if not param_name == 'constraints':
                # remember all parameters which are reverenced
                if 'datatype' not in param_config.keys():
                    warnings.warn(f"datatype not specified in {param_config} \
                                  for {param_name}, take float as default")
                    param_config['datatype'] = 'float'

                if 'reference' in param_config.keys():
                    referenced_params.update({param_name: param_config['reference']})
                # sample other parameter
                elif param_name != 'constraints':
                    dict_param_grids.update({param_name: sample_grid(param_config)})

        # create the grid from the individual parameter grids
        # constraints are not respected in this step
        grid_df_prior = pd.DataFrame(columns=['params'])
        # add shared parameters to dict_param_grids
        dict_param_grids = add_shared_params_to_param_grids(
            shared_df, dict_param_grids, config)
        add_next_param_from_list(dict_param_grids, {}, grid_df_prior)

        # add referenced params and check constraints
        add_references_and_check_constraints(grid_df_prior, grid_df, referenced_params,
                                             config, task_name)
        if grid_df[grid_df['algo'] == config['aname']].shape[0] == 0:
            raise RuntimeError('No valid value found for this grid spacing, refine grid')
        return grid_df
    elif 'shared' in config.keys():
        shared_grid = shared_df.copy()
        shared_grid['algo'] = config['aname']
        shared_grid['task'] = task_name
        if 'constraints' in config.keys():
            config['hyperparameters'] = {'constraints': config['constraints']}
        add_references_and_check_constraints(shared_grid, grid_df, {}, config, task_name)
        return grid_df
    else:
        # add single line if no varying hyperparameters are specified.
        grid_df.loc[len(grid_df.index)] = [task_name, config['aname'], {}]
        return grid_df


def sample_gridsearch(config: dict,
                      dest: str = None) -> pd.DataFrame:
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

    logger = Logger.get_logger()
    samples = pd.DataFrame(columns=['task', 'algo', 'params'])
    shared_samples_full = pd.DataFrame(columns=['task', 'algo', 'params'])

    if 'Shared params' in config.keys():
        shared_val = {'aname': 'all', 'hyperparameters':  config['Shared params']}
        # fill up the dataframe shared samples
        shared_samples_full = grid_task(shared_samples_full, 'all', shared_val, None)
    else:
        shared_samples_full = None
    for key, val in config.items():
        if sampling.is_dict_with_key(val, "aname"):
            if shared_samples_full is not None:
                shared_samples = shared_samples_full.copy(deep=True)
                if 'shared' in val.keys():
                    shared = val['shared']
                else:
                    shared = []
                for line_num in range(shared_samples.shape[0]):
                    hyper_p_dict = shared_samples.iloc[line_num]['params'].copy()
                    key_list = copy.deepcopy(list(hyper_p_dict.keys()))
                    if not all(x in key_list for x in shared):
                        raise RuntimeError(f"shared keys: {shared} not included in global shared keys {key_list}")
                    for key_ in key_list:
                        if key_ not in shared:
                            del hyper_p_dict[key_]
                    shared_samples.iloc[line_num]['params'] = hyper_p_dict
                # remove all duplicates
                shared_samples = shared_samples.drop_duplicates(subset='params')
            else:
                shared_samples = None

            samples = grid_task(samples, key, val, shared_samples)
            logger.info(f'number of gridpoints for {key} : '
                        f'{samples[samples["algo"] == val["aname"]].shape[0]}')

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    logger.info(f'number of total sampled gridpoints: {samples.shape[0]}')
    samples.to_csv(dest)
        # create a txt file with the commit information
    with open(config["output_dir"] + os.sep + 'commit.txt', 'w', encoding="utf8") as file:
        file.writelines("use git log |grep \n")
        file.writelines("consider remove leading b in the line below \n")
        file.write(get_git_tag())
    with open(config["output_dir"] + os.sep + 'config.txt', 'w', encoding="utf8") as file:
        json.dump(config, file)
    return samples
