"""
task specific dataset operation
"""
import random
from torch.utils.data import Dataset


class DsetIndDecorator4XYD(Dataset):
    """
    For dataset of x, y, d,  decorate it wih index
    """
    def __init__(self, dset):
        """
        :param dset: x,y,d
        """
        tuple_m = dset[0]
        if len(tuple_m) < 3:
            raise RuntimeError(
                "dataset to be wrapped should output at least x, y, and d; got length ",
                len(tuple_m))
        self.dset = dset

    def __getitem__(self, index):
        """
        :param index:
        """
        tensor_x, vec_y, vec_d, *_ = self.dset.__getitem__(index)
        return tensor_x, vec_y, vec_d, index

    def __len__(self):
        return self.dset.__len__()


class DsetZip(Dataset):
    """
    enable zip return in getitem: x_1, y_1, x_2, y_2
    to avoid always the same match, the second dataset does not use the same idx in __get__item()
    but instead, a random one
    """
    def __init__(self, dset1, dset2, name=None):
        """
        :param dset1: x1, y1, *d1
        :param dset2: x2, y2, *d2
        :param name: name of dataset
        """
        self.dset1 = dset1
        self.dset2 = dset2
        self.name = name
        self.len2 = self.dset2.__len__()

    def __getitem__(self, idx):
        """
        :param idx:
        """
        idx2 = idx + random.randrange(self.len2)
        idx2 = idx2 % self.len2
        tensor_x_1, vec_y_1, vec_d_1, *others_1 = self.dset1.__getitem__(idx)
        tensor_x_2, vec_y_2, vec_d_2, *others_2 = self.dset2.__getitem__(idx2)
        return tensor_x_1, vec_y_1, vec_d_1, others_1, tensor_x_2, vec_y_2, vec_d_2, others_2

    def __len__(self):
        len1 = self.dset1.__len__()
        if len1 < self.len2:
            return len1
        return self.len2
