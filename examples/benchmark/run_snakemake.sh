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

#snakemake -np -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml"
#snakemake --cores 1 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml" 2>&1 | tee $logfile
#snakemake --rerun-incomplete --cores 1 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml"


# print execution graph to pdf

#snakemake --dag --forceall -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml" | dot -Tpdf > dag.pdf

#snakemake --rulegraph --forceall -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml" | dot -Tpdf > rulegraph.pdf


# DENBI
#snakemake --keep-going --keep-incomplete --notemp --cores 4 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/test_denbi_benchmark.yaml" 2>&1 | tee $logfile
#snakemake --keep-going --keep-incomplete --notemp --cores 4 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/denbi_blood_benchmark.yaml" 2>&1 | tee $logfile


# DENBI MNIST
snakemake --keep-going --keep-incomplete --notemp --cores 1 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/denbi_mnist.yaml" 2>&1 | tee $logfile

