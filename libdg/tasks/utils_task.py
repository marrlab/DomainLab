"""
Task wraps around datasets, this file provide utilities
"""
import os
from pathlib import Path

import numpy
import torchvision
import torch
from torch.utils.data import Dataset
from libdg.utils.utils_class import store_args


class ImSize():
    """ImSize."""

    @store_args
    def __init__(self, i_c, i_h, i_w):
        """
        store channel, height, width
        """
    @property
    def c(self):
        """image channel"""
        return self.i_c

    @property
    def h(self):
        """image height"""
        return self.i_h

    @property
    def w(self):
        """image width"""
        return self.i_w


def mk_onehot(dim, ind):
    """
    :param dim: dimension of representation vector
    :param ind: index
    """
    eye = torch.eye(dim)
    vec = eye[ind]
    return vec


def mk_loader(dset, bsize, drop_last=True, shuffle=True):
    """
    :param bs: batch size
    """
    if len(dset) < bsize:
        bsize = len(dset)
    loader = torch.utils.data.DataLoader(
        dataset=dset,
        batch_size=bsize,
        shuffle=shuffle,
        # shuffle must be true so the last incomplete batch get used in anohter epoch
        num_workers=1,   # FIXME:
        drop_last=drop_last)
    return loader


class DsetDomainVecDecorator(Dataset):
    """
    decorate a pytorch dataset with a fixed vector representation of domain
    """
    def __init__(self, dset, vec_domain, na_domain):
        """
        :param dset: x, y
        :param vec_domain: vector representation of domain
        :param na_domain: string description of domain
        """
        self.dset = dset
        self.vec_domain = vec_domain
        self.na_domain = na_domain

    @property
    def targets(self):
        """
        return a list of all targets so class sample count is straight forward
        """
        return self.dset.targets

    def __getitem__(self, idx):
        """
        :param idx:
        """
        tensor, vec_class, *_ = self.dset.__getitem__(idx)
        return tensor, vec_class, self.vec_domain

    def __len__(self):
        """__len__."""
        return self.dset.__len__()


class DsetDomainVecDecoratorImgPath(DsetDomainVecDecorator):
    """
    Except returning x, y, d, additionally, the path of x is
    returned currently not in use since it is mostly important
    to print predictions together with path  for the test domain
    """
    def __getitem__(self, idx):
        """
        :param idx:
        """
        tensor, vec_class, path = self.dset.__getitem__(idx)
        return tensor, vec_class, self.vec_domain, path


class DsetClassVecDecorator(Dataset):
    """
    decorate a pytorch dataset with a new class name
    """
    def __init__(self, dset, dict_folder_name2class_global, list_str_y):
        """
        :param dset: x, y, *d
        :param dict_folder2class: dictionary that maps class folder of domain to glbal class
        """
        self.dset = dset
        self.class2idx = {k:v for (k,v) in self.dset.class_to_idx.items() if k in self.dset.list_class_dir}
        assert self.class2idx
        self.dict_folder_name2class_global = dict_folder_name2class_global
        self.list_str_y = list_str_y
        # inverst key:value to value:key for backward map
        self.dict_old_idx2old_class = dict((v, k) for k, v in self.class2idx.items())
        dict_class_na_local2vec_new = dict(
            (k, self.fun_class_local_na2vec_new(k)) for k, v in self.class2idx.items())
        self.dict_class_na_local2vec_new = dict_class_na_local2vec_new

    @property
    def targets(self):
        """
        return a list of all targets so class sample count is straight forward
        """
        return self.dset.targets

    def fun_class_local_na2vec_new(self, k):
        """
        local class name within one domain, to one-hot vector of new representation
        """
        ind = self.list_str_y.index(self.dict_folder_name2class_global[k])
        return mk_onehot(len(self.list_str_y), ind)

    def __getitem__(self, idx):
        """
        :param idx:
        """
        tensor, vec_class = self.dset.__getitem__(idx)
        vec_class = vec_class.numpy()
        ind_old = numpy.argmax(vec_class)
        class_local = self.dict_old_idx2old_class[ind_old]
        vec_class_new = self.dict_class_na_local2vec_new[class_local]
        return tensor, vec_class_new

    def __len__(self):
        """__len__."""
        return self.dset.__len__()


class DsetClassVecDecoratorImgPath(DsetClassVecDecorator):
    def __getitem__(self, idx):
        """
        :param idx:
        This function is mainly
        """
        tensor, vec_class, path = self.dset.__getitem__(idx)
        vec_class = vec_class.numpy()
        ind_old = numpy.argmax(vec_class)
        class_local = self.dict_old_idx2old_class[ind_old]
        vec_class_new = self.dict_class_na_local2vec_new[class_local]
        return tensor, vec_class_new, path


class LoaderDomainLabel():
    """
    wraps a dataset with domain label and into a loader
    """
    def __init__(self, batch_size, dim_d):
        """__init__.

        :param batch_size:
        :param dim_d:
        """
        self.batch_size = batch_size
        self.dim_d = dim_d

    def __call__(self, dset, d_ind, na_domain):
        """
        wrap_dataset2loader_with_domain_label.
        :param dataset:
        :param batch_size:
        :param d_dim:
        :param d_ind:
        """
        d_eye = torch.eye(self.dim_d)
        d_label = d_eye[d_ind]
        dset = DsetDomainVecDecorator(dset, d_label, na_domain)
        loader = mk_loader(dset, self.batch_size)
        return loader


def tensor1hot2ind(tensor_label):
    """tensor1hot2ind.

    :param tensor_label:
    """
    _, label_ind = torch.max(tensor_label, dim=1)
    npa_label_ind = label_ind.numpy()
    return npa_label_ind


def img_loader2dir(loader, folder, list_domain_na=None, list_class_na=None, batches=5):   # FIXME: this function couples strongly with the task, should be a class method of task
    """
    save images from loader to directory so speculate if loader is correct
    :param loader:
    :param folder:
    :param batches: default 1
    """
    Path(os.path.normpath(folder)).mkdir(parents=True, exist_ok=True)
    l_iter = iter(loader)
    counter = 0
    batches = min(batches, len(l_iter))
    for _ in range(batches):
        img, vec_y, *list_vec_domain = l_iter.next()
        class_label_ind_batch = tensor1hot2ind(vec_y)
        if list_vec_domain:  # if list is not empty
            domain_label_ind_batch = tensor1hot2ind(list_vec_domain[0])

        for b_ind in range(img.shape[0]):
            class_label_ind = class_label_ind_batch[b_ind]
            class_label_scalar = numpy.asscalar(class_label_ind)

            if list_class_na is None:
                str_class_label = "class_"+str(class_label_scalar)
            else:
                str_class_label = list_class_na[class_label_scalar]      # FIXME: where is the correspndance between class ind_label and class str_label?
            str_domain_label = "unknown"
            if list_vec_domain:
                domain_label_ind = domain_label_ind_batch[b_ind]
                if list_domain_na is None:
                    str_domain_label = str(domain_label_ind)
                else:
                    str_domain_label = list_domain_na[domain_label_ind]   # FIXME: the correspondance between domain ind_label and domain str_label is missing
            arr = img[b_ind]
            img_vision = torchvision.transforms.ToPILImage()(arr)
            f_n = "_".join(
                ["class", str_class_label, "domain", str_domain_label, "n", str(counter)])
            counter += 1
            path = os.path.join(folder, f_n + ".png")
            img_vision.save(path)
