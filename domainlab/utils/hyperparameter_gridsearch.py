import os

import numpy as np
import pandas as pd

from domainlab.utils.hyperparameter_sampling import is_task

def sample_cathegorical(settings):
    params = settings['values']
    return params

def sample_task(num_samples: int, sample_df: pd.DataFrame, task_name: str, config: dict):
    """Sample one task and add it to the dataframe"""
    algo = config['aname']
    if 'hyperparameters' in config.keys():
        constraints = config['hyperparameters'].get('constraints', None)
        for param in config['hyperparameters']:
            settings = config['hyperparameters'][param]
            if not 'hyperparameters' in config.keys():
                RuntimeError(f"the number of parameters in the grid direction "
                             f"of {param} needs to be specified")

            # sample cathegorical parameter
            if settings['distribution'] == 'categorical':
                grid_ = sample_cathegorical(settings)




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

    num_samples = config['num_param_samples']
    samples = pd.DataFrame(columns=['task', 'algo', 'params'])
    for key, val in config.items():
        if is_task(val):
            sample_task(num_samples, samples, key, val)

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    samples.to_csv(dest)
    return samples


