"""
https://github.com/pytorch/vision/blob/bb5af1d77658133af8be8c9b1a13139722315c3a/torchvision/datasets/folder.py#L93
https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html#DatasetFolder.fetch_img_paths
"""
import os
import sys
import warnings
from typing import Any, Tuple

from torchvision.datasets import DatasetFolder
from domainlab.utils.logger import Logger

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """
    Checks if a file is an allowed extension.
    Args:
    filename (string): path to a file
    extensions (tuple of strings): extensions to consider (lowercase)
    Returns: bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def fetch_img_paths(path_dir, class_to_idx, extensions=None, is_valid_file=None):
    """
    :param path_dir: path to fetch images in string format
    :param class_to_idx: given list of strings as class names
    {classes[i]: i for i in range(len(classes))}
    :param extensions: file extensions in fstring format
    :param is_valid_file: user provided function to check if the file is valid or not
    :return : list_tuple_path_cls_ind: list of tuple, (path of file, class index)
    """
    list_tuple_path_cls_ind = []
    path_dir = os.path.expanduser(path_dir)
    # since this function is only called by the class below, which now ensures that
    # extensions xor is_valid_file is not None, this check cannot be triggered
    # if not ((extensions is None) ^ (is_valid_file is None)):
    #     raise ValueError(
    #         "Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def functor_is_valid_file(filena):
            return has_file_allowed_extension(filena, extensions)
        is_valid_file = functor_is_valid_file
    for target in sorted(class_to_idx.keys()):
        apath = os.path.join(path_dir, target)
        if not os.path.isdir(apath):
            continue
        for root, _, fnames in sorted(os.walk(apath, followlinks=True)):
            for fname in sorted(fnames):
                path_file = os.path.join(root, fname)
                if is_valid_file(path_file):
                    item = (path_file, class_to_idx[target])
                    list_tuple_path_cls_ind.append(item)   # @FIXME
    return list_tuple_path_cls_ind


class DsetSubFolder(DatasetFolder):
    """
    Only use user provided class names, ignore the other subfolders
    :param list_class_dir: list of class directories to use as classes
    """
    def __init__(self, root, loader, list_class_dir, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        self.list_class_dir = list_class_dir

        if is_valid_file is not None and extensions is not None:
            raise ValueError(
                "Both extensions and is_valid_file cannot be not None at the same time")

        if is_valid_file is None and extensions is None:
            # setting default extensions
            extensions = ('jpg', 'jpeg', 'png')
            logger = Logger.get_logger()
            logger.warn("no user provided extensions, set to be jpg, jpeg, png")
            warnings.warn("no user provided extensions, set to be jpg, jpeg, png")

        super().__init__(root,
                         loader,
                         extensions=extensions,
                         transform=transform,
                         target_transform=target_transform,
                         is_valid_file=is_valid_file)
        classes, class_to_idx = self._find_classes(self.root)
        samples = fetch_img_paths(self.root, class_to_idx, extensions, is_valid_file)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.path2imgs = [s[0] for s in samples]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, (path,)

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
        logger = Logger.get_logger()
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            list_subfolders = [subfolder.name for subfolder in list(os.scandir(mdir))]
            logger.info(f"list of subfolders {list_subfolders}")
            classes = [d.name for d in os.scandir(mdir) \
                       if d.is_dir() and d.name in self.list_class_dir]
        else:
            classes = [d for d in os.listdir(mdir) \
                       if os.path.isdir(os.path.join(mdir, d)) and d in self.list_class_dir]
        flag_user_input_classes_in_folder = (set(self.list_class_dir) <= set(classes))
        if not flag_user_input_classes_in_folder:
            logger.info(f"user provided class names: {self.list_class_dir}")
            logger.info(f"subfolder names from folder: {mdir} {classes}")
            raise RuntimeError("user provided class names does not match the subfolder names")
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
