"""
probe task by saving images to folder with class and domain label
"""
from domainlab.compos.exp.exp_cuda_seed import set_seed
from domainlab.tasks.zoo_tasks import TaskChainNodeGetter
from domainlab.arg_parser import mk_parser_main


if __name__ == "__main__":
    parser = mk_parser_main()
    args = parser.parse_args()
    set_seed(args.seed)
    task = TaskChainNodeGetter(args)()
    task.init_business(args)
    task.sample_sav(root=args.out)
