#!/bin/bash

timestamp() {
#  date +"%T" # current time
  date +"%Y-%m-%d_%H-%M-%S"
}


logdir="zoutput/logs"
mkdir -p $logdir
logfile="$logdir/$(timestamp).out"
echo "verbose log: $logfile"

# -n: dry-run  (A dry run is a software testing process where the effects of a possible failure are intentionally mitigated, For example, there is rsync utility for transfer data over some interface, but user can try rsync with dry-run option to check syntax and test communication without data transferring.)
# -p: print shell commands
# -d: specify working directory. This should be the DomainLab dir
# -s: snakefile
# -- configfile: configuration yaml file of the benchmark

#snakemake -np -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml"
export DOMAINLAB_CUDA_START_SEED=$1

# first display all tasks
snakemake --cores 1 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml" --keep-going --summary  # this will give us a clue first what jobs will be run

# second submit the jobs
snakemake --cores 1 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml" --keep-going 2>&1 | tee "$logfile"

#snakemake --rerun-incomplete --cores 1 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml"

# print execution graph to pdf

#snakemake --dag --forceall -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml" | dot -Tpdf > dag.pdf

#snakemake --rulegraph --forceall -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml" | dot -Tpdf > rulegraph.pdf
