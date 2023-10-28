#!/bin/bash

timestamp() {
#  date +"%T" # current time
  date +"%Y-%m-%d_%H-%M-%S"
}

# CONFIGFILE="examples/yaml/test_helm_benchmark.yaml"
CONFIGFILE=$1
export DOMAINLAB_CUDA_START_SEED=$2

export DOMAINLAB_CUDA_HYPERPARAM_SEED=0

export NUMBER_GPUS=1
logdir="zoutput/logs"
mkdir -p $logdir
logfile="$logdir/$(timestamp).out"
echo "Configuration file: $CONFIGFILE"
echo "starting seed is: $DOMAINLAB_CUDA_START_SEED"
echo "verbose log: $logfile"
# Helmholtz
rm -f -R .snakemake
snakemake --profile "examples/yaml/slurm" --keep-going --keep-incomplete --notemp --cores 3 -s "domainlab/exp_protocol/benchmark.smk" --configfile "$CONFIGFILE" 2>&1 | tee "$logfile" 
