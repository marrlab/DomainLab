"""
https://github.com/pytorch/vision/blob/bb5af1d77658133af8be8c9b1a13139722315c3a/torchvision/datasets/folder.py#L93
https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html#DatasetFolder.fetch_img_paths
"""
import pytest

from domainlab.dsets.utils_data import fun_img_path_loader_default
from domainlab.dsets.dset_subfolder import DsetSubFolder


def test_fun():
    dset = DsetSubFolder(root="data/vlcs_mini/caltech",
                         list_class_dir=["auto", "vogel"],
                         loader=fun_img_path_loader_default,
                         extensions="jpg",
                         transform=None,
                         target_transform=None)
    dset.class_to_idx


def test_mixed_codec():
    dset = DsetSubFolder(root="data/mixed_codec/caltech",
                         list_class_dir=["auto", "vogel"],
                         loader=fun_img_path_loader_default,
                         extensions=None,
                         transform=None,
                         target_transform=None)
    assert len(dset.samples) == 6

    dset = DsetSubFolder(root="data/mixed_codec/caltech",
                         list_class_dir=["auto", "vogel"],
                         loader=fun_img_path_loader_default,
                         extensions="jpg",
                         transform=None,
                         target_transform=None)
    assert len(dset.samples) == 4, f"data/mixed_codec contains 4 jpg files, but {len(dset.samples)} were loaded."

    with pytest.raises(ValueError):
        DsetSubFolder(root="data/mixed_codec/caltech",
                      list_class_dir=["auto", "vogel"],
                      loader=fun_img_path_loader_default,
                      extensions="jpg",
                      transform=None,
                      target_transform=None,
                      is_valid_file=True)


def test_wrong_class_names():
    with pytest.raises(RuntimeError):
        DsetSubFolder(root="data/mixed_codec/caltech",
                      list_class_dir=["auto", "haus"],
                      loader=fun_img_path_loader_default,
                      extensions=None,
                      transform=None,
                      target_transform=None)
