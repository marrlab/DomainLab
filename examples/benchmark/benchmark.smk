import sys
from pathlib import Path

args = sys.argv
# get path to config file
# config_path = args[args.index("--configfile") + 1]
# if "-d" in args:
#     config_path = os.path.join(args[args.index("-d") + 1], config_path)
# elif "--directory" in args:
#     config_path = os.path.join(args[args.index("--directory") + 1],config_path)
try:
    config_path = workflow.configfiles[0]
except IndexError:
    raise RuntimeError("Please provide a config file using --configfile")



# TODO this seems a bit hacky and might break down.
#   is there a better option to make domainlab importable?
#   an easy-to-implement option would be to let the user specify the
#   domainlab path in the yaml file.
sys.path.insert(0, Path(workflow.basedir).parent.parent.as_posix())
# print(f"sys.path={sys.path}") # for debugging
# print(f"config_path={config_path}")


def experiment_result_files(_):
    """Lists all expected i.csv"""
    from domainlab.utils.hyperparameter_sampling import is_task
    # count tasks
    num_tasks = 0
    for key, val in config.items():
        if is_task(val):
            num_tasks += 1
    # total number of hyperparameter samples
    total_num_params = config['num_param_samples'] * num_tasks
    return [f"{config['output_dir']}/rule_results/{i}.csv" for i in range(total_num_params)]


rule all:
    input:
        # this changes of course to the plots
        expand("{output_dir}/results.csv", output_dir=config["output_dir"])
    default_target: True


rule parameter_sampling:
    input:
        # path to config file as input, thus a full
        # rerun is considered whenever the config yaml changed.
        expand("{path}", path=config_path)
    output:
        dest=expand("{output_dir}/hyperparameters.csv", output_dir=config["output_dir"])
    run:
        from domainlab.utils.hyperparameter_sampling import sample_hyperparameters
        sample_hyperparameters(config, str(output.dest))


# TODO continue after breakdown works, but jobs with incomplete seed
#   iteration are not restarted, but only the partial results are included
#   at the results.csv.
rule run_experiment:
    input:
        param_file=expand("{output_dir}/hyperparameters.csv", output_dir=config["output_dir"])
    output:
        out_file=temporary(expand(
            "{output_dir}/rule_results/{index}.csv",
            output_dir=config["output_dir"],
            allow_missing=True
        ))
    run:
        from domainlab.exp_protocol.run_experiment import run_experiment
        index = int(expand(wildcards.index)[0])
        run_experiment(config, str(input.param_file), index, str(output.out_file), config['domains'])


rule agg_results:
    input:
        param_file=expand("{output_dir}/hyperparameters.csv",output_dir=config["output_dir"]),
        exp_results=experiment_result_files
    output:
        out_file=expand("{output_dir}/results.csv", output_dir=config["output_dir"])
    run:
        import os
        out_file = str(output.out_file)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        has_header = False
        # print(f"exp_results={input.exp_results}")
        with open(out_file, 'w') as out_stream:
            for res in input.exp_results:
                with open(res, 'r') as in_stream:
                    if has_header:
                        # skip header line
                        in_stream.readline()
                    else:
                        out_stream.writelines(in_stream.readline())
                        has_header = True
                    # write results to common file.
                    out_stream.writelines(in_stream.readlines())


# TODO
#   rule agg_partial_results:

# TODO
#   rule gen_plots:
