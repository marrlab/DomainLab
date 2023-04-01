# Benchmarking with DomainLab

The package offers the ability to benchmark different algorithms against each other.
The results are collected in a csv file, but also prepared in charts.

Within each benchmark, two aspects are considered:
1. Stochastic variation: Variation of the performance with respect to different random seeds.
2. Sensitivity to selected hyperparameters: By sampling hyperparameters randomly,
the performance with respect to different hyperparameter choices is investigated. 

## Setting up a benchmark
The benchmark is configured in a yaml file. We refer to the
[demo_benchmark.yaml](https://github.com/marrlab/DomainLab/blob/master/examples/yaml/demo_benchmark.yaml)
for a documented example. As one can see here, the user has to select:
- Common arguments for all tasks under `domainlab_args`. This typically includes the dataset on which the benchmark
shall be performed, as well as the number of epochs and batch size for training.
- The number of different random seeds and hyperparameter samples is set.
- The domains which are used as leave-one-out for testing.
- Different tasks, where each task contains: (1) Fixed parameters specific for this task,
typically the algorithm name and related arguments and (2) varying hyperparameters, for which
the sensitivity shall be investigated.

Set `sampling_seed` for a fully reproducible benchmark.

## Configure hyperparameter sampling
An example files of the possibilities for hyperparameter sampling is
[demo_hypeparameter_sampling.yml](https://github.com/marrlab/DomainLab/blob/master/examples/yaml/demo_hyperparameter_sampling.yml).
This file is of course an extract of [demo_benchmark.yaml](https://github.com/marrlab/DomainLab/blob/master/examples/yaml/demo_benchmark.yaml),
the sampling for benchmarks is specified directly in this single configuration file.

In this example we can see:
- Every node containing `aname` is considered as a task
- Hyperparameters can be sampled from different distributions and
- Constraints between the hyperparameters of the same task can be specified.
- By using the `categorical` distribution type, a list of valid values for a hyperparameter can be specified.
- Equality constraints are not supported in the constraints section. To
enforce equality of two hyperparameters use the `reference` key, see `p4` of `Task1`.
References are supported only to sampled hyperparameters, referencing a reference
results in undefined behaviour.

Note that for sampling, there is always first a sample drawn ignoring the constraints, which
is then possibly rejected by the constraints. If 10,000 samples are rejected in a row, the sampling
aborts with error.

## Running a benchmark
For executing the benchmark, a sample command can be found
[here](https://github.com/marrlab/DomainLab/blob/benchmark_snakemake/examples/benchmark/run_snakemake.sh).
If several cores are provided, the benchmark is parallelized per hyperparameter sample and
task.

To run the benchmark with a specific configuration, one can execute 

```
./run_benchmark.sh ./examples/yaml/demo_benchmark.yaml 0 
```
where the first argument is the benchmark configuration file and the second argument is optional which is the starting seed.

## Obtained results
All files created by this benchmark are saved in the given output directory.
The sampled hyperparameters can be found in `hyperparameters.csv`.
The performance of the different runs can be found in `results.csv`. Moreover, there is
the `graphics` subdirectory, in which the values from `results.csv` are visualized for interpretation.


## Obtain partial results
The results form partially completed benchmarks can be obtained with
```commandline
python main_out.py --agg_partial_bm OUTPUT_DIR
```
specifying the benchmark output directory, e.g. `zoutput/benchmarks/demo_benchmark`
