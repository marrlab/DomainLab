"""
Test the benchmark plots using some dummy results saved in .csv files
"""
from domainlab.utils.generate_benchmark_plots import gen_benchmark_plots


def test_benchm_plots():
    """
    test benchmark plots
    """
    gen_benchmark_plots(
        "domainlab/zdata/ztest_files/aggret_res_test1",
        "zoutput/benchmark_plots_test/outp1",
        use_param_index=False,
    )
    gen_benchmark_plots(
        "domainlab/zdata/ztest_files/aggret_res_test2", "zoutput/benchmark_plots_test/outp2"
    )
