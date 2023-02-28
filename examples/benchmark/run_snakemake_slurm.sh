#!/bin/bash

timestamp() {
#  date +"%T" # current time
  date +"%Y-%m-%d_%H-%M-%S"
}

export DOMAINLAB_CUDA_START_SEED=$1
logdir="zoutput/logs"
mkdir -p $logdir
logfile="$logdir/$(timestamp).out"
echo "verbose log: $logfile"

# Helmholtz
snakemake --profile "examples/yaml/slurm" --keep-going --keep-incomplete --notemp --cores 5 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/test_helm_benchmark.yaml" 2>&1 | tee $logfile 
