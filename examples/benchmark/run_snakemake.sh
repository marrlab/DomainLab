echo $PWD
# -n: dry-run
# -p: print shell commands
# -d: specify working directory. This should be the DomainLab dir
# -s: snakefile
# -- configfile: configuration yaml file of the benchmark

#snakemake -np -s "examples/benchmark/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml"
snakemake --cores 1 -s "examples/benchmark/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml"
#snakemake --rerun-incomplete --cores 1 -s "examples/benchmark/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml"


# print execution graph to pdf
#snakemake --dag --forceall -s "examples/benchmark/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml" | dot -Tpdf > dag.pdf
#snakemake --rulegraph --forceall -s "examples/benchmark/benchmark.smk" --configfile "examples/yaml/demo_benchmark.yaml" | dot -Tpdf > rulegraph.pdf
