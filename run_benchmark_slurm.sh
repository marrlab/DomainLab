#!/bin/bash

# Source the common functions script
source scripts/common_benchmark_functions.sh

# CONFIGFILE="examples/yaml/test_helm_benchmark.yaml"
CONFIGFILE=$1
logfile=$(create_log_file)
echo "Configuration file: $CONFIGFILE"
echo "verbose log: $logfile"

# Configuring DOMAINLAB_CUDA_START_SEED
DOMAINLAB_CUDA_START_SEED=${2:-0}
export DOMAINLAB_CUDA_START_SEED
echo "argument 2: DOMAINLAB_CUDA_START_SEED=$DOMAINLAB_CUDA_START_SEED"

# ensure all runs sample the same hyperparameters
export DOMAINLAB_CUDA_HYPERPARAM_SEED=0

export NUMBER_GPUS=1

results_dir=$(extract_output_dir "$CONFIGFILE")

echo "Starting seed is: $DOMAINLAB_CUDA_START_SEED"
echo "Hyperparameter seed is: $DOMAINLAB_CUDA_HYPERPARAM_SEED"
echo "Number of GPUs: $NUMBER_GPUS"
echo "Results will be stored in: $results_dir"

# Helmholtz
snakemake --profile "examples/yaml/slurm" --config yaml_file="$CONFIGFILE" --keep-going --keep-incomplete --notemp --cores 3 -s "domainlab/exp_protocol/benchmark.smk" --configfile "$CONFIGFILE" --config output_dir="$results_dir" 2>&1 | tee "$logfile"
