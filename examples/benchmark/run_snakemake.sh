#!/bin/sh
# -n: dry-run  (A dry run is a software testing process where the effects of a possible failure are intentionally mitigated, For example, there is rsync utility for transfer data over some interface, but user can try rsync with dry-run option to check syntax and test communication without data transferring.)
# -p: print shell commands
# -d: specify working directory. This should be the DomainLab dir
# -s: snakefile
# -- configfile: configuration yaml file of the benchmark

#snakemake -np -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml"
export DOMAINLAB_CUDA_START_SEED=$1

snakemake --cores 1 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml" --keep-going --summary  # this will give us a clue first what jobs will be run

snakemake --cores 1 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml" --keep-going

#snakemake --rerun-incomplete --cores 1 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml"


## Print execution graph to pdf ##

#snakemake --dag --forceall -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml" | dot -Tpdf > dag.pdf

#snakemake --rulegraph --forceall -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml" | dot -Tpdf > rulegraph.pdf

## Run snakemake with slurm  ##

#snakemake --profile "/home/icb/xinyue.zhang/DomainLab/examples/yaml/slurm" --keep-going --keep-incomplete --notemp --cores 5 -s "domainlab/exp_protocol/benchmark.smk" --configfile "examples/yaml/test_helm_benchmark_debug.yaml" 2>&1 | tee $logfile 