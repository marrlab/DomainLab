"""
https://github.com/pytorch/vision/blob/bb5af1d77658133af8be8c9b1a13139722315c3a/torchvision/datasets/folder.py#L93
https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html#DatasetFolder.fetch_img_paths
"""
import os
import sys
from typing import cast, Callable, Tuple, Any
from torchvision.datasets import DatasetFolder
from domainlab.dsets.utils_data import fun_img_path_loader_default
from domainlab.dsets.dset_subfolder import DsetSubFolder


def test_fun():
    dset = DsetSubFolder(root="zdpath/vlcs_small_class_mismatch/caltech",
                         list_class_dir=["auto", "vogel"],
                         loader=fun_img_path_loader_default,
                         extensions="jpg",
                         transform=None,
                         target_transform=None)
    dset.class_to_idx
