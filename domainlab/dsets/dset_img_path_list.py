import os

import torch.utils.data as data

from domainlab.dsets.utils_data import fun_img_path_loader_default
from domainlab.utils.utils_class import store_args


class DsetImPathList(data.Dataset):
    @store_args
    def __init__(self, root_img, path2filelist, trans_img=None, trans_target=None):
        """
        one file provide image path and label which forms a domain
        """
        self.list_tuple_img_label = []
        self.get_list_tuple_img_label()

    def get_list_tuple_img_label(self):
        with open(self.path2filelist, 'r') as f_h:
            for str_line in f_h.readlines():
                path_img, label_img = str_line.strip().split()
                self.list_tuple_img_label.append((path_img, int(label_img)))   # @FIXME: string to int, not necessarily continuous

    def __getitem__(self, index):
        path_img, target = self.list_tuple_img_label[index]
        target = target - 1   # @FIXME:  make this more general
        img = fun_img_path_loader_default(os.path.join(self.root_img, path_img))
        if self.trans_img is not None:
            img = self.trans_img(img)
        if self.trans_target is not None:
            target = self.trans_target(target)
        return img, target, path_img

    def __len__(self):
        return len(self.list_tuple_img_label)
