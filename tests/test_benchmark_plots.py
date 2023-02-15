'''
Test the benchmark plots using some dummy results saved in .csv files
'''
from domainlab.utils.generate_benchmark_plots import gen_benchmark_plots


def test_benchm_plots():
    '''
    test benchmark plots
    '''
    gen_benchmark_plots('data/ztest_files/aggret_res_test1', 'outp1')
    gen_benchmark_plots('data/ztest_files/aggret_res_test2', 'outp2')
