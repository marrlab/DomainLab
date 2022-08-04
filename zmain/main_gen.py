"""
command line generate images
"""
import os
import torch
from domainlab.compos.exp.exp_cuda_seed import set_seed
from domainlab.tasks.zoo_tasks import TaskChainNodeGetter
from domainlab.arg_parser import mk_parser_main
from domainlab.utils.utils_cuda import get_device
from domainlab.utils.flows_gen_img_model import fun_gen


def main_gen(args, task=None, model=None, device=None):
    """
    command line generate images
    """
    device = get_device(args.nocu)
    node = TaskChainNodeGetter(args)()
    node.init_business(args)
    model = torch.load(args.mpath, map_location="cpu")
    model = model.to(device)
    subfolder_na = os.path.basename(args.mpath)
    fun_gen(model, device, node, args, subfolder_na)


if __name__ == "__main__":
    parser = mk_parser_main()
    parser.add_argument('--mpath', type=str, default=None,
                        help="path for persisted model")
    args = parser.parse_args()
    set_seed(args.seed)
    main_gen(args)
