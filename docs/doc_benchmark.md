# Benchmarking with DomainLab

[Documentation for Benchmark in Markdown](https://github.com/marrlab/DomainLab/blob/master/docs/doc_benchmark.md)

The package offers the ability to benchmark different user-defined experiments against each other,
as well as against different hyperparameter settings and random seeds.
The results are collected in a csv file, but also prepared in charts.

Within each benchmark, two aspects are considered:
1. Stochastic variation: variation of the performance with respect to different random seeds.
2. Sensitivity to selected hyperparameters: by sampling hyperparameters randomly,
the performance with respect to different hyperparameter choices is investigated.

## Setting up a benchmark
The benchmark is configured in a yaml file. We refer to [doc_benchmark_yaml.md](https://github.com/marrlab/DomainLab/blob/master/docs/doc_benchmark_yaml.md) for a documented
example. 

## Running a benchmark
For the execution of a benchmark we provide two scripts in our repository:
- local version for running the benchmark on a standalone machine:
[run_benchmark_standalone.sh](https://github.com/marrlab/DomainLab/blob/master/run_benchmark_standalone.sh)
- cluster version for running the benchmark on a slurm cluster: [run_benchmark_slurm.sh](https://github.com/marrlab/DomainLab/blob/master/run_benchmark_slurm.sh)

### Benchmark on a standalone machine (with or without GPU)
To run the benchmark with a specific configuration on a standalone machine, inside the DomainLab 
folder, one can execute (we assume you have a machine with 4 cores or more)
```shell
# Note: this has only been tested on Linux based systems and may not work on Windows
./run_benchmark_standalone.sh ./examples/benchmark/demo_benchmark.yaml 0  0  2
```
where the first argument is the benchmark configuration file (mandatory), the second and the third 
arguments are the starting seeds for cuda and the hyperparameter sampling (both optional) and the
fourth argument is the number of GPUs to use (optional). The number of GPUs defaults to one 
(if your machine does not have GPU, the last argument defaults to one as well and CPU is used).

In case of snakemake error, try
`rm -r .snakemake/`

### Benchmark on a HPC cluster with slurm
If you have access to an HPC cluster with slurm support: In a submission node, clone the DomainLab
repository, cd into the repository and execute the following command:

**Make sure to use tool like nohup or tmux to keep the following command active!**

It is a good idea to use standalone script to test if the yaml file work or not before submit to slurm cluster. 

```cluster
# Note: this has only been tested on Linux based systems and may not work on Windows
./run_benchmark_slurm.sh ./examples/benchmark/demo_benchmark.yaml
```
Similar to the local version explained above, the user can also specify a random seed for 
hyperparameter sampling and pytorch.

#### Check errors for slurm runs
The following script will help to find out which job has failed and the error message, so that you could direct to the 
specific log file
```cluster
bash ./sh_list_error.sh ./zoutput/slurm_logs
```
#### Map between slurm job id and sampled hyperparameter index
suppose the slurm job id is 14144163, one could the corresponding log file in `./zoutput/slurm_logs` folder via
`find . | grep -i "14144163"`

the results can be
`run_experiment-index=41-14144163.err`
where `41` is the hyperparameter index in `zoutput/benchmarks/[name of the benchmark]/hyperparameters.csv`. 


## Obtained results
All files created by this benchmark are saved in the given output directory
(by default `./zoutput/benchmarks/[name of the benchmark defined in YAML file]`). The sampled hyperparameters can be found in
`hyperparameters.csv`. The yaml file is translated to `config.txt` with corresponding to commit in formation in `commit.txt` (**do not update code during benchmark process so results can be reproducible with this commit information**), corresponding to each line in `hyperparameters.csv`, there will
be a csv file in directory `rule_results`.

#### Output folder structure

via `tree -L 2` in `zoutput/benchmarks/[name of the benchmark defined in configuration yaml file]`, one can get something like below

```                                                
├── commit.txt
├── config.txt
├── graphics
│   ├── diva_fbopt_full
│   ├── radar_dist.png
│   ├── radar.png
│   ├── scatterpl
│   ├── sp_matrix_dist.png
│   ├── sp_matrix_dist_reg.png
│   ├── sp_matrix.png
│   ├── sp_matrix_reg.png
│   └── variational_plots
├── hyperparameters.csv
├── results.csv
└── rule_results
    ├── 0.csv
    ├── 1.csv
    ├── 2.csv
    ├── 3.csv
    ├── 4.csv
    ├── 5.csv
    ├── 6.csv
    └── 7.csv
```
where commit.txt contains commit information for reproducibility, config.txt is a json format of the configuration yaml file for reproducibility, graphics folder contains the visualization of benchmark results in various plots, specificly, we use `graphics/variational_plot/acc/stochastic_variation.png`, hyperparameters.csv contains all hyperparameters used for each method, results.csv is an aggregation of the csv files in `rule_results`, where the i.csv correspond to the parameter index in hyperparameters.csv

**Please do not change anything in folder `rule_results` !**

The performance of the different runs from directory `rule_results` will be aggregated after all jobs have been done, which can be found aggregated in `results.csv`, Moreover, there is the `graphics` subdirectory, in which the values from `results.csv` are
visualized for interpretation.

In case that the benchmark is not entirely completed, the user can obtain partial results as
explained below.


### Obtain partial results
If the benchmark is not yet completed (still running or has some failed jobs), the `results.csv` file containing the aggregated results will not be created.
The user can then obtain the aggregated partial results with plots from the partially completed benchmark by running
the following after cd into the DomainLab directory:
```commandline
python main_out.py --agg_partial_bm OUTPUT_DIR
```
specifying the benchmark output directory containing the partially completed benchmark,
e.g. `./zoutput/benchmarks/demo_benchmark`, where `demo_benchmark` is a name defined in the yaml file. 

Alternatively, one could use 
```examples
cat ./zoutput/benchmarks/[name of the benchmark]/rule_results/*.csv > result.csv
```
clean up the extra csv head generated and plot the csv using command below

### Generate plots from .csv file
If the benchmark is not completed, the `graphics` subdirectory might not be created. The user can then manually
create the graphics from the csv file of the aggregated partial results, which can be obtained as explained above.
Here for, the user must cd into the DomainLab directory and run

```commandline
python main_out.py --gen_plots CSV_FILE --outp_dir OUTPUT_DIR
```

specifying the path of the csv file of the aggregated results (e.g. `./zoutput/benchmarks/demo_benchmark/results.csv`)
and the output directory of the partially completed benchmark (e.g. `./zoutput/benchmarks/demo_benchmark`).
Note that the cvs file must have the same form as the one generated by the fully executed benchmark, i.e. 





| param_index | task | algo | epos | te_d | seed | params | acc | precision | recall | specificity | f1 | auroc | 
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ... | ... | ... | ... | ... | {'param1': p1, ...} | ... | ... | ... | ... | ... | ... |
| 1 | ... | ... | ... | ... | ... | {'param1': p2, ...} | ... | ... | ... | ... | ... | ... |
