# Benchmarking with DomainLab

The package offers the ability to benchmark different user-defined experiments against each other,
as well as against different hyperparameter settings and random seeds.
The results are collected in a csv file, but also prepared in charts.

Within each benchmark, two aspects are considered:
1. Stochastic variation: variation of the performance with respect to different random seeds.
2. Sensitivity to selected hyperparameters: by sampling hyperparameters randomly,
the performance with respect to different hyperparameter choices is investigated.

## Setting up a benchmark
The benchmark is configured in a yaml file. We refer to [demo_benchmark.yaml](https://github.com/marrlab/DomainLab/blob/master/examples/benchmark/demo_benchmark.yaml) for a documented
example. The user can create different custom experiments, which are to be benchmarked. Each
experiment can have a custom name.

An experiment can specify following arguments:
- `aname`: name of the model. An experiment with an incorrect `aname` will not be considered in the
benchmark! (mandatory)
- `hyperparameters`: model-specific hyperparameters (for more details on the hyperparameters we
refer to the model-specific documentation). The hyperparameters can also be randomly sampled, see
next section
- `shared`: a list of the parameter, which the respective experiment shares with another experiment
(optional)
- `constraints`: a list of constraints, see section below (optional)
- `trainer`: trainer to be used (optional, e.g. "dial" or "mldg")
- model-specific arguments (for more details we refer to the model-specific documentation)

Furthermore, the user can declare:
- `mode`: technique for hyperparameter sampling (for details see next section)
- `domainlab_args`: common arguments for all benchmarking experiments, including:
  - `tpath`/`task`: path to/name of the task, which should be addressed (mandatory)
  - dataset on which the benchmark is performed
  - `epos`: number of epochs (int, mandatory)
  - `bs`: batch size for training (int, mandatory)
  - `lr`: learning rate (float, mandatory)
  - `es`: early stop steps (int, optional)
  - `npath`/`nname`: path to/name of the neural network, which should be used for feature extraction
    (mandatory, can also be specified in each single experiment)
  - `dmem`: `True` or `False` (optional)
  - `san_check`: `True`or `False` (optional)
  - `te_d`: list of test domains (mandatory)
  - `tr_d`: list of training domains (mandatory)
- `output_dir`: path to the custom output directory (mandatory)
- `num_param_samples`: number of hyperparameters to be sampled (int, mandatory)
- `Shared params`: a list including `num_shared_param_samples` (number of samples for the shared
params) and the shared hyperparameters with respective sampling distribution (optional)

Depending on which hyperparameter sampling technique is used (see section below), the user must also
respect/declare the following:
- random hyperparameter sampling:
  - `sampling_seed` (int), `startseed` (int), `endseed` (int) must be defined outside the experiment
  - `distribution`: specifies the distribution used for sampling, must be specified for each
  hyperparameter (for available distributions see section below)
  definitions (mandatory)
- grid search hyperparameter sampling (`grid`):
  - `num`: number of hyperparameters to be sampled (int) must be specified for each hyperparameter
  (mandatory)
  - `distribution`: specifies the distribution used for sampling, must be specified for each
  hyperparameter (for available distributions see section below)
  

### Hyperparameter sampling
The package offers the option to randomly sample hyperparameters from different distributions.
An example can be found in [demo_hyperparameter_sampling.yml](https://github.com/marrlab/DomainLab/blob/master/examples/yaml/demo_hyperparameter_sampling.yml). We offer two sampling
techniques, random hyperparameter sampling and grid search. Each technique offers the following
distributions to sample from:
- `categorical` distribution. For each parameter the user can specify:
  - `values`: a list of valid values 
  - `datatype`: the datatype of the list elements
- `uniform` and `loguniform` distribution. The user must define the following for each
hyperparameter (mandatory):
  - `mean`: mean of normal distribution (float)
  - `std`: standard deviation of normal distribution (float >= 0)
- `normal` and `lognormal`distribution. The user must define the following for each hyperparameter
  (mandatory):
  - `min`: lower bound for samples (int)
  - `max`: upper bound for samples (int)
    - `step`: "step-size" (float) of grid starting from lower bound. Only points lying on the grid
    can be sampled. `0` means that each real number represents a grid point and thus, can be sampled. 
  

### Constraints
The user can specify a list of constraints for the hyperparameters. Please note the following:
- We currently use rejection sampling, to prevent e.g. the case of contradictory constraints,
amongst others. In concrete, the sampling aborts with an error if 10.000 samples are rejected in a
row. 
- Equality constraints are not supported in the constraints section. To enforce equality of two or 
more hyperparameters use the `reference` key, see `p4` of `Task1` in
[demo_hypeparameter_sampling.yml](https://github.com/marrlab/DomainLab/blob/master/examples/yaml/demo_hyperparameter_sampling.yml). References are supported only to sampled hyperparameters,
referencing a reference results in undefined behaviour.


## Running a benchmark
For the execution of a benchmark we provide two scripts in our repository:
- local version for running the benchmark on a standalone machine:
[run_benchmark_local_conf_seed2_gpus.sh](https://github.com/marrlab/DomainLab/blob/master/run_benchmark_local_conf_seed2_gpus.sh)
- cluster version for running the benchmark on a slurm cluster: [run_benchmark_slurm.sh](https://github.com/marrlab/DomainLab/blob/master/run_benchmark_slurm_conf_seed2.sh)

### benchmark on a standalone machine (with or without GPU)
To run the benchmark with a specific configuration on a standalone machine, inside the DomainLab 
folder, one can execute (we assume you have a machine with 4 cores or more)
```shell
# Note: this has only been tested on Linux based systems and may not work on Windows
./run_benchmark_local_conf_seed2_gpus.sh ./examples/benchmark/demo_benchmark.yaml 0  0
```
where the first argument is the benchmark configuration file, and the second and third argument,
which is the starting seed for cuda and hyperparameter sampling, is optional. By the last optional
argument the user can also specify the number of GPUs to use, by default it is one, and this should
be the case for cpu as well (if your machine does not have GPU, the last argument will be set to 1
as well).

### benchmark on a HPC cluster with slurm
If you have access a HPC cluster with slurm support: In a submission node, clone the DomaniLab
repository, cd into the repository and execute the following command.
```cluster
# Note: this has only been tested on Linux based systems and may not work on Windows
./run_benchmark_slurm_conf_seed2.sh ./examples/benchmark/demo_benchmark.yaml
```
Similar to the local version, one could also specify random seed for hyperparameter sampling and
random seed for pytorch.

## Obtained results
All files created by this benchmark are saved in the given output directory
(by default `./zoutput/benchmarks`). The sampled hyperparameters can be found in
`hyperparameters.csv`. The performance of the different runs can be found in `results.csv`.
Moreover, there is the `graphics` subdirectory, in which the values from `results.csv` are
visualized for interpretation.


## Obtain partial results
The results from partially completed benchmarks can be obtained with
```commandline
python main_out.py --agg_partial_bm OUTPUT_DIR
```
specifying the benchmark output directory, e.g. `./zoutput/benchmarks/demo_benchmark`.

## Generate plots from .csv file

For generating the graphics directly from a csv file the user can run 

```commandline
python main_out.py --gen_plots CSV_FILE --outp_dir OUTPUT_DIR
```

Note that the cvs file must have the same form as the ones generated by the benchmark into
results.csv, i.e. 





| param_index | task | algo | epos | te_d | seed | params | acc | precision | recall | specificity | f1 | auroc | 
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ... | ... | ... | ... | ... | {'param1': p1, ...} | ... | ... | ... | ... | ... | ... |
| 1 | ... | ... | ... | ... | ... | {'param1': p2, ...} | ... | ... | ... | ... | ... | ... |

