"""
Utilities for dataset
"""
import datetime

import torch
import torch.utils.data as data_utils
from PIL import Image
from torch.utils.data import Dataset
from torchvision.utils import save_image
from domainlab.utils.logger import Logger


def fun_img_path_loader_default(path):
    """
    https://discuss.pytorch.org/t/handling-rgba-images/88428/4
    """
    return Image.open(path).convert('RGB')


def mk_fun_label2onehot(dim):
    """
    function generator
    index to onehot
    """
    def fun_label2onehot(label):
        """
        :param label:
        """
        m_eye = torch.eye(dim)
        return m_eye[label]
    return fun_label2onehot


def plot_ds(dset, f_name, batchsize=32):
    """
    :param dset:
    :param f_name:
    :param batchsize: batch_size
    """
    loader_tr = data_utils.DataLoader(dset, batch_size=batchsize, shuffle=False)
    for _, (img, _, *_) in enumerate(loader_tr):
        nrow = min(img.size(0), 8)
        save_image(img.cpu(), f_name, nrow=nrow)
        break  # only one batch


def plot_ds_list(ds_list, f_name, batchsize=8, shuffle=False):
    """
    plot list of datasets, each datasets in one row
    :param ds_list:
    :param fname:
    :param batchsize:
    :param shuffle:
    """
    list_imgs = []
    for dset in ds_list:
        loader = data_utils.DataLoader(dset, batch_size=batchsize, shuffle=shuffle)
        for _, (img, _, *_) in enumerate(loader):
            list_imgs.append(img)
            break
    comparison = torch.cat(list_imgs)
    save_image(comparison.cpu(), f_name, nrow=batchsize)


class DsetInMemDecorator(Dataset):
    """
    fetch all items of a dataset into memory
    """
    def __init__(self, dset, name=None):
        """
        :param dset: x, y, *d
        :param name: name of dataset
        """
        self.dset = dset
        self.item_list = []
        logger = Logger.get_logger()
        if name is not None:
            logger.info(f"loading dset {name}")
        t_0 = datetime.datetime.now()
        for i in range(len(self.dset)):
            self.item_list.append(self.dset[i])
        t_1 = datetime.datetime.now()
        logger.info(f"loading dataset to memory taken: {t_1 - t_0}")

    def __getitem__(self, idx):
        """
        :param idx:
        """
        return self.item_list[idx]

    def __len__(self):
        return self.dset.__len__()
