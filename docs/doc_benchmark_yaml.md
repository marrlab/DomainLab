# Benchmark yaml files

yaml files are a powerful tool to specify the details of a benchmark.
The following examples on how a yaml file could look like will lead you through constructing your own yaml file for a benchmark.

We will start with the general setting of the yaml file and then continue with the description of how to define the sampling/gridsearch for the hyperparameters.

## General setup of the yaml file

One may start with a very general setup of the file, defining all fixed information like task description, test and training domains, used algorithms ...

At the top level, we need to decide whether the random sampling or grid search shall be used.

For random samples we need to define the total number of hyperparameter samples in each sampling task (`num_param_samples`) and a sampling seed for the hyperparameters (`sampling_seed`).

For grid search the mode needs to be switched to grid using `mode: grid`, the specification on how many samples sould be created in gridsearch will be done in the next section when setting up the parameter ranges and distributions.

```yaml
# output_dir defines the output directory to be used to save the results
output_dir: zoutput/benchmarks/mnist_benchmark

# test domains all algorithms are tested on
test_domains:
  - 0
  - 1

##### For random sampling #####
# for random sampling you can exactly specify the number of samples you like to use
# per algorithm (4 algorithms with 8 hyperparameter samples each -> 4 * 8 combinations)
num_param_samples: 8
# setting a seed for the random sampling
sampling_seed: 0

##### For grid search #####
# define the mode to be grid search
mode: grid


# in each run the network is initialised randomly, therefore it might happen, that
# the results are different, even though the same hyperparameters were used
# to test for this it is possible to run the same hyperparameter setting multiple times
# with different random seeds used to initialise the network.
# "endseed - startseed + 1" experiments are run with the same hyperparameter sample
startseed: 0
endseed: 5  # currently included


###################################################################################
# Arguments in the section domainlab_args are passed to all tasks.
# Task specific tasks take precedence.
domainlab_args:
    # task specification, this could also be a task path
    # (i.e. for pacs dataset tpath: examples/tasks/task_pacs_path_list.py)
    task: mnistcolor10

    # training domain for all algorithms
    tr_d:
        - 2
        - 3

    # learning rate
    lr: 0.001

    # maximal number of epochs for each run
    epos: 50

    # number of iterations without process before early stopping is applied
    es: 5

    # batch size
    bs: 64

    # specification of the network to be used, this could also be a network path
    # (i.e. for resnet50, npath: examples/nets/resnet50domainbed.py)
    nname: conv_bn_pool_2

    # some of the algorithms do need multiple networks,
    # all of which can also be paths to a network
    # (i.e. npath_... : examples/nets/resnet50domainbed.py)
    nname_encoder_x2topic_h: conv_bn_pool_2
    nname_encoder_sandwich_x2h4zd: conv_bn_pool_2
    nname_dom: conv_bn_pool_2


###################################################################################
# Hyperparameters which appear in multiple tasks can e shared among these tasks.
# Hence for each task the same random sampled hyperparameters are used
Shared params:
    #### For random sampling #####
    # number of shared samples to be created.
    # The sampling procedure creates a set of randomly sampled shared samples,
    # each algorithm which uses one of the shared samples will randomly pick its
    # sample from this set.
    num_shared_param_samples: 8
    # gridsearch will crash if num_shared_param_samples is set

    # shared hyperparameters:
    <<ADD SAMPLING DESCRIPTIONS HERE>>

    # ... you may like to add more shared samples here like:
    # gamma_y, gamma_d, zy_dim, zd_dim


###################################################################################
################################## TASKS ##########################################
###################################################################################
# From here on we start defining the different tasks, starting each section with
# a unique task name

Task_Diva_Dial:
    # set the method to be used, if model is skipped the Task will not be executed
    model: diva

    # select a trainer to be used, if trainer is skipped adam is used
    # options: "dial" or "mldg"
    trainer: dial

    # Here we can also set task specific hyperparameters
    # which shall be fixed among all experiments.
    # f not set, the default values will be used.
    zd_dim: 32
    zx_dim: 64
    gamma_d: 1e5

    # define which hyperparameters from the shared section shall be used.
    # In this task gamma_reg, zx_dim, zy_dim, zd_dim are used.
    shared:
        - gamma_reg
        - zx_dim
        - zy_dim
        - zd_dim

    # define task specific hyperparameter sampling
    hyperparameters:
        <<ADD SAMPLING DESCRIPTIONS HERE>>

        # add constraints for your sampled hyperparameters,
        # by using theire name in a python expression.
        # You can use all hyperparameters defined in the hyperparameter section of
        # the current task and the shared hyperparameters specified in the shared
        # section of the current task
        constraints:
        - 'zx_dim <= zy_dim'


# add more tasks
Task_Jigen:
    ...

Task_Dann:
    ...
```

## Sampling description

There are two possible ways of choosing your hyperparameters for the benchmark in domainlab, rand sampling and grid search. The decision about which method to use was already done in the previous section be either setting `num_param_samples` (for random sampling) or `mode: grid` (for gridsearch).

For filling in the sampling description for the into the `Shared params` and the `hyperparameter` section you have the following options:

### uniform and loguniform distribution
1. uniform samples in the interval [min, max]
```yaml
tau:                            # name of the hyperparameter
    min: 0.01
    max: 1
    distribution: uniform       # name of the distribution
    ##### for grid search #####
    num: 3                      # number of grid points created for this hyperparameter
```

2. loguniform samples in the interval [min, max]. This is usefull if the interval spans over multiple magnitudes.
```yaml
gamma_y:                        # name of the hyperparameter
    min: 1e4
    max: 1e6
    distribution: loguniform    # name of the distribution
    ##### for grid search #####
    num: 3                      # number of grid points created for this hyperparameter
```

### normal and lognormal distribution
1. normal samples with mean and standard deviation
```yaml
pperm:                          # name of the hyperparameter
    mean: 0.5
    std: 0.2
    distribution: normal        # name of the distribution
    ##### for grid search #####
    num: 3                      # number of grid points created for this hyperparameter
```

2. lognormal samples with mean and standard deviation. This is usefull if the interval spans over multiple magnitudes.
```yaml
gamma_y:                        # name of the hyperparameter
    mean: 1e5
    std: 2e4
    distribution: loguniform    # name of the distribution
    ##### for grid search #####
    num: 3                      # number of grid points created for this hyperparameter
```

### cathegorical hyperparameters

choose the values of the hyperparameter from a predefined list. If one uses grid search, then all values from the list are used as grid points
```yaml
nperm:                          # name of the hyperparameter
    distribution: categorical   # name of the distribution
    datatype: int
    values:                     # concrete values to choose from
      - 30
      - 31
      - 100
```

### Referenced hperparameters

If one hyperparameter does directly depend on another hyperparameter you can add a reference to the other parameter. For gridsearch num will be taken from the reference, therefore it cannot be specified here.
```yaml
zy_dim:                         # name of the hyperparameter
    reference: 2 * zx_dim       # formular to be evaluated in python
```

### Special Arguments in the Sampling description
1. **datatype:** specify the datatype of the samples, if `int`, the values are rounded to the next integer. This works will all distributions mentioned above.
    - datatypes can be set to be `int`, othervise they will be `float`
```yaml
zx_dim:                         # name of the hyperparameter
    min: 0
    max: 96
    distribution: uniform
    datatype: int               # dimesions must be integer values
    ##### for grid search #####
    num: 3                      # number of grid points created for this hyperparameter
```
2. **step:** if thep is specified the values are only chosen from a grid with grid size `step`. for random sampling the algorithm will randomly sample from this grid. For gridsearch the algorithm will create a subgrid of the grid with grid size `step`.
```yaml
zx_dim:                         # name of the hyperparameter
    min: 0
    max: 96
    distribution: uniform
    step: 16                    # only choose from values {0, 16, 32, 84, 64, 80, 96}
    datatype: int               # dimesions must be integer values
    ##### for grid search #####
    num: 3                      # number of grid points created for this hyperparameter
```

## Combination of Shared and Task Specific Hyperparameter Samples

it is possible to have all sorts of combinations:
1. a task which includes shared and task specific sampled hyperparameters
```yaml
Task_Name:
    model: ...
    ...

    # specify sections from the Shared params section
    shared:
        - ...
    # specify task specific hyperparameter sampling
    hyperparameters:
        ...
        # add the constraints to the hperparameters section
        constraints:
        - '...'     # constraints using params from the hyperparameters and the shared section
```

2. Only task specific sampled hyperparameters
```yaml
Task_Name:
    model: ...
    ...

    # specify task specific hyperparameter sampling
    hyperparameters:
        ...
        # add the constraints to the hperparameters section
        constraints:
        - '...'     # constraints using only hyperparameters from the hyperparameters section
```

3. Only shared sampled hyperparamters
```yaml
Task_Name:
    model: ...
    ...

    # specify sections from the Shared params section
    shared:
        - ...
    # add the constraints as a standalone section to the task
    constraints:
    - '...'         # constraints using only hyperparameters from the shared section
```

4. No hyperparameter sampling. All Hyperparameters are either fixed to a user defined value or to the default value. No hyperparameter samples indicates no constraints.
```yaml
Task_Name:
    model: ...
    ...
```
