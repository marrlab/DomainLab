"""
The class TaskPathList provide the user an interface to provide a file with
each line consistiing of a pair, where the first slot contains the path
(either absolute or relative if the user knows from where this package is executed)
of an image and the second slot contains the label of numerical string.
"""
import os
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
# "too many opened files" https://github.com/pytorch/pytorch/issues/11201

from torchvision import transforms
from domainlab.tasks.b_task import NodeTaskDict
from domainlab.dsets.utils_data import mk_fun_label2onehot
from domainlab.dsets.dset_img_path_list import DsetImPathList


class NodeTaskPathListDummy(NodeTaskDict):
    pass


def mk_node_task_path_list(isize,
                           list_str_y,
                           trans4all,
                           dict_class_label2name,
                           dict_domain2imgroot,
                           dict_d2filepath_list_img,
                           dict_d2filepath_list_img_val,
                           dict_d2filepath_list_img_te,
                           succ=None):
    """mk_node_task_path_list.

    :param isize:
    :param list_str_y:
    :param trans4all:
    :param dict_class_label2name:
    :param dict_domain2imgroot:
    :param dict_d2filepath_list_img:
    :param dict_d2filepath_list_img_val:
    :param dict_d2filepath_list_img_te:
    :param succ:
    """
    class NodeTaskPathList(NodeTaskPathListDummy):
        """
        The class TaskPathList provide the user an interface to provide a file
        with each line consisting of a pair separated by comma, where the
        first slot contains the path (either absolute or relative if the user
        knows from where this package is executed) of an image and the second slot
        contains the class label as a numerical string.
        e.g.: /path/2/file/art_painting/dog/pic_376.jpg 1
        """
        def _get_complete_domain(self, na_domain, list_domain_path):
            """_get_complete_domain.

            :param na_domain:
            """
            if self._dict_domain_img_trans:
                trans = self._dict_domain_img_trans[na_domain]
            else:
                trans = trans4all
            root_img = self.dict_domain2imgroot[na_domain]
            path2filelist = list_domain_path[na_domain]
            path2filelist = os.path.expanduser(path2filelist)
            root_img = os.path.expanduser(root_img)
            dset = DsetImPathList(root_img, path2filelist, trans_img=trans,
                                  trans_target=mk_fun_label2onehot(
                                      len(self.list_str_y)))
            return dset

        def get_dset_by_domain(self, args, na_domain, split=True):
            """get_dset_by_domain.

            :param args:
            :param na_domain:
            :param split: for test set, use the whole
            """
            if not split:  # no train/val split for test domain
                dset = self._get_complete_domain(
                    na_domain,
                    self._dict_domain2filepath_list_im_te)
                return dset, dset  # FIXME: avoid returning two identical

            dset = self._get_complete_domain(
                na_domain,
                self._dict_domain2filepath_list_im)

            dset_val = self._get_complete_domain(
                na_domain,
                self._dict_domain2filepath_list_im_val)

            return dset, dset_val

        def conf(self, args):
            """conf.

            :param args:
            """
            self.list_str_y = list_str_y
            self.isize = isize
            self.dict_domain2imgroot = dict_domain2imgroot
            self._dict_domain2filepath_list_im = dict_d2filepath_list_img
            self._dict_domain2filepath_list_im_val = dict_d2filepath_list_img_val
            self._dict_domain2filepath_list_im_te = dict_d2filepath_list_img_te
            self.set_list_domains(list(self.dict_domain2imgroot.keys()))

        def init_business(self, args):
            """init_business.

            :param args:
            """
            self.conf(args)
            super().init_business(args)

    return NodeTaskPathList(succ)
