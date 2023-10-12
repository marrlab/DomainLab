## Further explanations to Benchmark Setup
The user can create different custom experiments, which are to be benchmarked. Each
experiment can have a custom name.

An experiment can specify following arguments:
- `aname`: name of the model. An experiment with an incorrect `aname` is not considered in the
benchmark (mandatory)
- `hyperparameters`: model-specific hyperparameters (for more details on the hyperparameters we
refer to the model-specific documentation). The hyperparameters can be randomly sampled, see
next section (optional)
- `shared`: a list of parameters, which the respective experiment shares with another experiment
(optional)
- `constraints`: a list of constraints for the hyperparameters, see section below (optional)
- `trainer`: trainer to be used (optional, e.g. "dial" or "mldg")
- model-specific arguments (for more details we refer to the model-specific documentation)

Furthermore, the user can declare:
- `domainlab_args`: common arguments for all benchmarking experiments, including:
  - `tpath`/`task`: path to/name of the task which should be addressed (mandatory)
  - `epos`: number of epochs (int, mandatory)
  - `bs`: batch size for training (int, mandatory)
  - `lr`: learning rate (float, mandatory)
  - `es`: early stop steps (int, optional)
  - `npath`/`nname`: path to/name of the neural network, which should be used for feature extraction
    (mandatory, can also be specified within each experiment)
  - `dmem`: `True` or `False` (optional)
  - `san_check`: `True`or `False` (optional)
  - `te_d`: list of test domains (mandatory)
  - `tr_d`: list of training domains (mandatory)
- `output_dir`: path to the custom output directory (mandatory)
- `startseed`: creates reproducible results (mandatory)
- `endseed`: creates reproducible results (mandatory)
- `mode`: set to `grid` to apply grid search for hyperparameter sampling (optional, for details see next section)
- `Shared params`: an optional list including the shared hyperparameters with respective sampling distribution 
(mandatory if Shared params should be used) and in case of random sampling `num_shared_param_samples` 
(number of samples for the shared hyperparameters, mandatory for random sampling) 

Depending on which hyperparameter sampling technique is used (see section below), the user must also
respect/declare the following:
- random hyperparameter sampling:
  - `sampling_seed` (int) must be defined outside the experiments
  - `num_param_samples`: number of hyperparameters to be sampled (int, mandatory), must be defined outside the experiments
  - `distribution`: specifies the distribution used for sampling, must be specified for each
  hyperparameter, for available distributions see section below (mandatory)
  - `step`: "step-size" (float) between samples. Only points being a multiple of the step-size apart
  can be sampled. `0` means that each real number can be sampled.
  - `num_shared_param_samples`: number of samples for the shared hyperparameters. Must be defined 
  inside the Shared params section (if this section is used)
- grid search hyperparameter sampling (`mode`:`grid`):
  - `num`: number of hyperparameters to be sampled (int) must be specified for each hyperparameter
  (mandatory)
  - `distribution`: specifies the distribution used for sampling, must be specified for each
  hyperparameter (for available distributions see section below)
  - `step`: "step-size" (float) of the grid points. Only points lying on the grid
  can be sampled. `0` means that each real number represents a grid point and thus, can be sampled. 
  

### Hyperparameter sampling
The benchmark offers the option to randomly sample hyperparameters from different distributions.
An example can be found in [demo_hyperparameter_sampling.yml](https://github.com/marrlab/DomainLab/blob/master/examples/yaml/demo_hyperparameter_sampling.yml). We offer two sampling
techniques, random hyperparameter sampling and grid search. The default sampling mode is random
hyperparameter sampling. If grid search should be applied, the user must specify `mode`:`grid`.

Each sampling technique offers the following distributions:
- `categorical` distribution. For each parameter the user can specify:
  - `values`: a list of valid values 
  - `datatype`: the datatype of the list values (int or float, default: float)
- `uniform` and `loguniform` distribution. The user must define the following for each
hyperparameter (mandatory):
  - `mean`: mean of normal distribution (float)
  - `std`: standard deviation of normal distribution (float $\geq 0$)
  - `datatype`: the datatype of the list values (int or float, default: float)
- `normal` and `lognormal`distribution. The user must define the following for each hyperparameter
  (mandatory):
  - `min`: lower bound for samples (int)
  - `max`: upper bound for samples (int)
  - `datatype`: the datatype of the list values (int or float, default: float)
  
  

### Constraints
The user can specify a list of constraints for the hyperparameters. Please note the following:
- We currently use rejection sampling, to prevent the case of contradictory constraints,
amongst others. In concrete, the sampling aborts with an error if 10.000 samples are rejected in a
row. 
- Equality constraints are not supported in the constraints section. To enforce equality of two or 
more hyperparameters use the `reference` key, see `p4` of `Task1` in
[demo_hypeparameter_sampling.yml](https://github.com/marrlab/DomainLab/blob/master/examples/yaml/demo_hyperparameter_sampling.yml). References are only supported to reference sampled hyperparameters, e.g.
referencing a reference results in undefined behaviour.
