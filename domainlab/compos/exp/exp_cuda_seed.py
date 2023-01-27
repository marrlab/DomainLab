"""
Random seed should be set from command line to ensure reproducibility:
https://pytorch.org/docs/stable/notes/randomness.html
https://discuss.pytorch.org/t/difference-between-torch-manual-seed-and-torch-cuda-manual-seed/13848/6
"""
import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # for reproducibility, this has to be False: benchmark mode is good
    # whenever your input sizes for your network do not vary. This way,
    # cudnn will look for the optimal set of algorithms for that particular
    # configuration (which takes some time). This usually leads to faster
    # runtime.
    np.random.seed(int(seed))
