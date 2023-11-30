from domainlab.exp.exp_main import Exp
from domainlab.exp.exp_cuda_seed import set_seed  # reproducibility
from domainlab.arg_parser import parse_cmd_args

if __name__ == "__main__":
    args = parse_cmd_args()
    set_seed(args.seed)
    exp = Exp(args=args)
    exp.execute()
