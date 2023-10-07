import os
import sys
from pathlib import Path
import pandas as pd


try:
    config_path = workflow.configfiles[0]
except IndexError:
    raise RuntimeError("Please provide a config file using --configfile")



# NOTE: this approach to obtain the path depends on the relative path of
# this file to the domainlab directory

sys.path.insert(0, Path(workflow.basedir).parent.parent.as_posix())

envvars:
    "DOMAINLAB_CUDA_START_SEED",
    "DOMAINLAB_CUDA_HYPERPARAM_SEED",
    "NUMBER_GPUS"


def experiment_result_files(_):
    """Lists all expected i.csv"""
    from domainlab.utils.hyperparameter_sampling import is_dict_with_key
    from domainlab.utils.logger import Logger
    from domainlab.utils.hyperparameter_gridsearch import \
        sample_gridsearch

    logger = Logger.get_logger()
    if 'mode' in config.keys():
        if config['mode'] == 'grid':
            # hyperparameters are sampled using gridsearch
            # in this case we don't know how many samples we will get beforehand
            # straigt oreward solution: do a grid sampling and count samples
            samples = sample_gridsearch(config)
            total_num_params = samples.shape[0]
            logger.info(f"total_num_params={total_num_params} for gridsearch")
    else:
        # in case of random sampling it is possible to compute the number
        # of samples from the information in the yaml file

        # count tasks
        num_sample_tasks = 0
        num_nonsample_tasks = 0
        for key, val in config.items():
            if is_dict_with_key(val, "aname"):
                if 'hyperparameters' in val.keys():
                    num_sample_tasks += 1
                else:
                    if 'shared' in val.keys():
                        num_sample_tasks += 1
                    else:
                        num_nonsample_tasks += 1
        # total number of hyperparameter samples
        total_num_params = config['num_param_samples'] * num_sample_tasks + num_nonsample_tasks
        logger.info(f"total_num_params={total_num_params} for random sampling")
        logger.info(f"={config['num_param_samples']} * {num_sample_tasks} + {num_nonsample_tasks}")

    return [f"{config['output_dir']}/rule_results/{i}.csv" for i in range(total_num_params)]


rule parameter_sampling:
    input:
        # path to config file as input, thus a full
        # rerun is considered whenever the config yaml changed.
        expand("{path}", path=config_path)
    output:
        dest=expand("{output_dir}/hyperparameters.csv", output_dir=config["output_dir"])
    params:
        sampling_seed=os.environ["DOMAINLAB_CUDA_HYPERPARAM_SEED"]
    run:
        from domainlab.utils.hyperparameter_sampling import sample_hyperparameters
        from domainlab.utils.hyperparameter_gridsearch import sample_gridsearch

        # for gridsearch there is no random component, therefore no
        # random seed is needed
        if 'mode' in config.keys():  # type(config)=dict
            if config['mode'] == 'grid':
                sample_gridsearch(config,str(output.dest))
        # for random sampling we need to consider a random seed
        else:
            sampling_seed_str = params.sampling_seed
            if isinstance(sampling_seed_str, str) and (len(sampling_seed_str) > 0):
              # hash will keep integer intact and hash strings to random seed
              # hased integer is signed and usually too big, random seed only
              # allowed to be in [0, 2^32-1]
              # if the user input is number, then hash will not change the value,
              # so we recommend the user to use number as start seed
              if sampling_seed_str.isdigit():
                sampling_seed = int(sampling_seed_str)
              else:
                sampling_seed = abs(hash(sampling_seed_str)) % (2 ** 32)
            elif 'sampling_seed' in config.keys():
              sampling_seed = config['sampling_seed']
            else:
              sampling_seed = None

            sample_hyperparameters(config, str(output.dest), sampling_seed)


rule run_experiment:
    input:
        param_file=rules.parameter_sampling.output
    output:
        # snakemake keyword temporary for temporary directory
        # like f-string in python {index} is generated in the run block as wildcards
        out_file=temporary(expand(
            "{output_dir}/rule_results/{index}.csv",
            output_dir=config["output_dir"],
            allow_missing=True
        ))
    params:
        start_seed_str=os.environ["DOMAINLAB_CUDA_START_SEED"]
    resources:
        nvidia_gpu=1
    run:
        from domainlab.exp_protocol.run_experiment import run_experiment
        # import sys
        # pos = None
        # try:
        #  pos = sys.argv.index('--envvars')
        # except Exception as ex:
        #  pos = None
        # start_seed = sys.argv[pos+1]
        num_gpus_str=os.environ["NUMBER_GPUS"]
        start_seed_str = params.start_seed_str
        if isinstance(start_seed_str, str) and (len(start_seed_str) > 0):
          # hash will keep integer intact and hash strings to random seed
          # hased integer is signed and usually too big, random seed only
          # allowed to be in [0, 2^32-1]
          # if the user input is number, then hash will not change the value,
          # so we recommend the user to use number as start seed
          if start_seed_str.isdigit():
            start_seed = int(start_seed_str)
          else:
            start_seed = abs(hash(start_seed_str)) % (2 ** 32)
        else:
          start_seed = None # use start seed defined in benchmark yaml configuration file
        # {index} defines wildcards named index
        index = int(expand(wildcards.index)[0])
        # :param config: dictionary from the benchmark yaml
        # :param param_file: path to the csv with the parameter samples
        # :param param_index: parameter index that should be covered by this task
        # currently this correspond to the line number in the csv file, or row number
        # in the resulting pandas dataframe
        # :param out_file: path to the output csv
        num_gpus = int(num_gpus_str)
        run_experiment(config, str(input.param_file), index,str(output.out_file),
            start_seed, num_gpus=num_gpus)


rule agg_results:
    # put different csv file in a big csv file
    input:
        exp_results=experiment_result_files
    output:
        out_file=expand("{output_dir}/results.csv", output_dir=config["output_dir"])
    run:
        from domainlab.exp_protocol.aggregate_results import agg_results
        agg_results(list(input.exp_results), str(output.out_file))


rule gen_plots:
    # depends on previous rules of agg_(partial_)results
    input:
        res_file=rules.agg_results.output.out_file
    output:
        out_dir=directory(expand("{output_dir}/graphics", output_dir=config["output_dir"]))
    run:
        from domainlab.utils.generate_benchmark_plots import gen_benchmark_plots
        gen_benchmark_plots(str(input.res_file), str(output.out_dir))


rule all:
    # output of plotting generation as input, i.e. all previous stages have to be carried out
    input:
        rules.gen_plots.output
    default_target: True
