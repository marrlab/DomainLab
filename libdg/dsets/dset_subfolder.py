"""
https://github.com/pytorch/vision/blob/bb5af1d77658133af8be8c9b1a13139722315c3a/torchvision/datasets/folder.py#L93
https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html#DatasetFolder.make_dataset
"""
import os
import sys
from typing import cast, Callable, Tuple
from torchvision.datasets import DatasetFolder


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """
    Checks if a file is an allowed extension.
    Args:
    filename (string): path to a file
    extensions (tuple of strings): extensions to consider (lowercase)
    Returns: bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)   # FIXME

    return images


class DsetSubFolder(DatasetFolder):
    """
    Only use user provided class names, ignore the other subfolders
    :param list_class_dir: list of class directories to use as classes
    """
    def __init__(self, root, loader, list_class_dir, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        self.list_class_dir = list_class_dir
        def fun_is_valid_file(input):
            return True   # FIXME
        if is_valid_file is None:
            is_valid_file = cast(Callable[[str], bool], fun_is_valid_file)
            super().__init__(root, loader, extensions=None, transform=transform,   # FIXME:extension
                         target_transform=target_transform, is_valid_file=is_valid_file)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, None, is_valid_file)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, mdir):
        """
        Finds the class folders in a dataset.
        Args:
            mdir (string): Root mdirectory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (mdir),
            and class_to_idx is a dictionary.
        Ensures:
            No class is a submdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            list_subfolders = [subfolder.name for subfolder in list(os.scandir(mdir))]
            print("list of subfolders", list_subfolders)
            classes = [d.name for d in os.scandir(mdir) \
                       if d.is_dir() and d.name in self.list_class_dir]
        else:
            classes = [d for d in os.listdir(mdir) \
                       if os.path.isdir(os.path.join(mdir, d)) and d in self.list_class_dir]
        flag_user_input_classes_in_folder = (set(self.list_class_dir) <= set(classes))
        if not flag_user_input_classes_in_folder:
            print("user provided class names:", self.list_class_dir)
            print("subfolder names from folder:", mdir, classes)
            raise RuntimeError("user provided class names does not match the subfolder names")
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


def test_fun():
    from libdg.dsets.utils_data import fun_img_path_loader_default
    dset = DsetSubFolder(root="zdpath/vlcs_small_class_mismatch/caltech",
                         list_class_dir=["auto", "vogel"],
                         loader=fun_img_path_loader_default,
                         extensions="jpg",
                         transform=None,
                         target_transform=None)
    dset.class_to_idx
