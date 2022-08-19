import os
import datetime
from domainlab.tasks.zoo_tasks import TaskChainNodeGetter
from domainlab.compos.exp.exp_utils import AggWriter
from domainlab.algos.zoo_algos import AlgoBuilderChainNodeGetter
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # debug
from domainlab.utils.arg_parser import mk_parser_main
from domainlab.compos.exp.exp_main import Exp

def test_exp():
    parser = mk_parser_main()
    args = parser.parse_args(["--te_d", "2", "--task", "mnistcolor10", "--debug"])
    exp = Exp(args)
    exp.execute()
