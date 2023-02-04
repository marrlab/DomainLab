
rule parameter_sampling:
    # TODO would be cool to have {wildcard} for the yml filename and
    #   benchmarks subdir here.
    input:
        "examples/yaml/benchmark.yml"
    output:
        "zoutput/benchmarks/benchmark/hyperparameters.csv"
    shell:
        "python "