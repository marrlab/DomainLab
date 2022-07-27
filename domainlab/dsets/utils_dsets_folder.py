import os
from torch.utils.data.dataset import ConcatDataset
from torchvision import transforms
from torchvision.datasets import DatasetFolder

from domainlab.dsets.utils_data import mk_fun_label2onehot, fun_img_path_loader_default
from domainlab.utils.utils_class import store_args


class DsetFolderGetter(object):
    """
    prefix_tr="train",
    prefix_val="crossval",
    prefix_te="test",
    prefix_full="full",
    domain_dict={'CALTECH': 0,
                'LABELME': 1,
                'PASCAL': 2,
                'SUN': 3},
    """
    @store_args
    def __init__(self, raw_dir,
                 extensions="jpg",
                 class_dict={'bird': 0,
                             'car': 1,
                             'chair': 2,
                             'dog': 3,
                             'human': 4},
                 transf_tr=transforms.ToTensor(),
                 transf_te=transforms.ToTensor(),
                 img_path_loader=fun_img_path_loader_default):
        """
        specify the folder structure
        """
        self.class_num = len(class_dict)

    def set_transform(self, transf_tr, transf_te):
        self.transf_tr = transf_tr
        self.transf_te = transf_te

    def __call__(self, na_split_folder=None, na_domain=None):
        if na_split_folder is not None and na_domain is not None:
            path = os.path.join(self.raw_dir, na_domain, na_split_folder)
        elif na_split_folder is not None:
            path = os.path.join(self.raw_dir, na_split_folder)
        else:
            path = os.path.normpath(self.raw_dir)
        dset = DatasetFolder(root=path,
                             loader=self.img_path_loader,
                             extensions=self.extensions,
                             transform=self.transf_tr,
                             target_transform=mk_fun_label2onehot(self.class_num))

        return dset



def test_vlcs():
    dug = DsetFolderGetter("~/Documents/dataset/vlcs/VLCS")
    dset = dug("CALTECH")
    dset[0]
    len(dset)
    dset.samples
    len(set(dset.samples))
    dset_f = dug("full", "CALTECH")
    len(dset_f)
    dset_tr = dug("train", "CALTECH")
    len(dset_tr)
    dset_te = dug("test", "CALTECH")
    len(dset_te)
    dset_val = dug("crossval", "CALTECH")
    len(dset_val)
    len(dset_val)+len(dset_te)+len(dset_tr) == len(dset_f)
    len(dset_val)+len(dset_te)+len(dset_tr) + len(dset_f) == len(dset)


    dug2 = DsetFolderGetter("~/Documents/dataset/vlcs_overlap_class/")
    dset_c = dug2("full", "CALTECH")
    dset_c.classes
    dset_c.class_to_idx
    dset_l = dug2("train", "LABELME")
    dset_l.class_to_idx
