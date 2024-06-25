#!/bin/bash

set -e

# Source the common functions script
source scripts/sh_benchmark_utils.sh

CONFIGFILE=$1
logfile=$(create_log_file)
echo "Configuration file: $CONFIGFILE"
echo "verbose log: $logfile"

# Configuring DOMAINLAB_CUDA_START_SEED
DOMAINLAB_CUDA_START_SEED=${2:-0}
export DOMAINLAB_CUDA_START_SEED
echo "argument 2: DOMAINLAB_CUDA_START_SEED=$DOMAINLAB_CUDA_START_SEED"

# Configuring DOMAINLAB_CUDA_HYPERPARAM_SEED
DOMAINLAB_CUDA_HYPERPARAM_SEED=${3:-0}
export DOMAINLAB_CUDA_HYPERPARAM_SEED
echo "argument 3: DOMAINLAB_CUDA_HYPERPARAM_SEED=$DOMAINLAB_CUDA_HYPERPARAM_SEED"

echo "argument 4: NUMBER_GPUS=$4"
NUMBER_GPUS=${4:-1}
export NUMBER_GPUS
echo "argument 4: NUMBER_GPUS set to ${NUMBER_GPUS}"

results_dir=$(extract_output_dir "$CONFIGFILE")

echo "Starting seed is: $DOMAINLAB_CUDA_START_SEED"
echo "Hyperparameter seed is: $DOMAINLAB_CUDA_HYPERPARAM_SEED"
echo "Number of GPUs: $NUMBER_GPUS"
echo "Results will be stored in: $results_dir"

# -n: dry-run  (A dry run is a software testing process where the effects of a possible failure are intentionally mitigated, For example, there is rsync utility for transfer data over some interface, but user can try rsync with dry-run option to check syntax and test communication without data transferring.)
# -p: print shell commands
# -d: specify working directory. This should be the DomainLab dir
# -s: snakefile
# -- configfile: configuration yaml file of the benchmark


# first display all tasks
snakemake --rerun-incomplete --cores 1 -s "domainlab/exp_protocol/benchmark.smk" --configfile "$CONFIGFILE" --keep-going --summary  # this will give us a clue first what jobs will be run

# second submit the jobs, make sure you have more than 4 cores on your laptop, otherwise adjust the cores
snakemake --config yaml_file="$CONFIGFILE" --rerun-incomplete --resources nvidia_gpu="$NUMBER_GPUS" --cores 4 -s "domainlab/exp_protocol/benchmark.smk" --configfile "$CONFIGFILE" --config output_dir="$results_dir" 2>&1 | tee "$logfile"

# snakemake --rerun-incomplete --cores 1 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml"
# snakemake -np -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml"


# print execution graph to pdf

# snakemake --dag --forceall -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml" | dot -Tpdf > dag.pdf

# snakemake --rulegraph --forceall -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml" | dot -Tpdf > rulegraph.pdf

# Command used to run in the DENBI cluster
# snakemake --keep-going --keep-incomplete --notemp --cores 5 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/helm_runtime_evaluation.yaml" 2>&1 | tee $logfile

# Command used to run in the Helmholtz cluster
# snakemake --profile "examples/yaml/slurm" --keep-going --keep-incomplete --notemp --cores 5 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/test_helm_benchmark.yaml" 2>&1 | tee "$logfile"

# Command used to run snakemake on a demo benchmark
# snakemake -np -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml"

# First display all tasks
# snakemake --cores 1 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml" --keep-going --summary  # this will give us a clue first what jobs will be run

# second submit the jobs
# snakemake --cores 1 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml" --keep-going 2>&1 | tee "$logfile"

# snakemake --rerun-incomplete --cores 1 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml"

# only plot
# snakemake --cores 1 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml" --keep-going --allowed-rules agg_partial_results
# snakemake --cores 1 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml" --keep-going --allowed-rules gen_plots
