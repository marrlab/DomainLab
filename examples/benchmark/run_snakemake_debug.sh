#!/bin/bash

timestamp() {
#  date +"%T" # current time
  date +"%Y-%m-%d_%H-%M-%S"
}


logdir="zoutput/logs"
mkdir -p $logdir
logfile="$logdir/$(timestamp).out"
echo "verbose log: $logfile"

# -n: dry-run
# -p: print shell commands
# -d: specify working directory. This should be the DomainLab dir
# -s: snakefile
# -- configfile: configuration yaml file of the benchmark



# DENBI
#snakemake --keep-going --keep-incomplete --notemp --cores 5 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/helm_runtime_evaluation.yaml" 2>&1 | tee $logfile
# TODO create a snakemake call (alike above) to run on the helmholz vm
# TODO set a reasonable number of cores (cores = 5 was very usefull, as we start 5 algorithms -> each algo gets one core)

# Helmholtz
snakemake --profile "/home/icb/xinyue.zhang/DomainLab/examples/yaml/slurm" --keep-going --keep-incomplete --notemp --cores 5 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/test_helm_benchmark_debug.yaml" 2>&1 | tee $logfile 
