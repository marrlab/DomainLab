"""
The class TaskPathList provides the user an interface to provide a file with
each line consisting of a pair, where the first slot contains the path
(either absolute or relative if the user knows from where this package is
executed)
of an image and the second slot contains the class label as a numerical string.
"""
import os

import torch.multiprocessing

from domainlab.dsets.dset_img_path_list import DsetImPathList
from domainlab.dsets.utils_data import mk_fun_label2onehot
from domainlab.tasks.b_task_classif import NodeTaskDictClassif

torch.multiprocessing.set_sharing_strategy('file_system')
# "too many opened files" https://github.com/pytorch/pytorch/issues/11201


class NodeTaskPathListDummy(NodeTaskDictClassif):
    """
    typedef class so that other function can use isinstance
    """
    def get_dset_by_domain(self, args, na_domain, split=False):
        raise NotImplementedError


def mk_node_task_path_list(isize,
                           img_trans_te,
                           list_str_y,
                           img_trans_tr,
                           dict_class_label_ind2name,
                           dict_domain2imgroot,
                           dict_d2filepath_list_img_tr,
                           dict_d2filepath_list_img_val,
                           dict_d2filepath_list_img_te,
                           succ=None):
    """mk_node_task_path_list.

    :param isize:
    :param list_str_y:
    :param img_trans_tr:
    :param dict_class_label_ind2name:
    :param dict_domain2imgroot:
    :param dict_d2filepath_list_img_tr:
    :param dict_d2filepath_list_img_val:
    :param dict_d2filepath_list_img_te:
    :param succ:
    """
    class NodeTaskPathList(NodeTaskPathListDummy):
        """
        The class TaskPathList provides the user an interface to provide a file
        with each line consisting of a pair separated by comma, where the
        first slot contains the path (either absolute or relative if the user
        knows from where this package is executed) of an image and the second
        slot contains the class label as a numerical string.
        e.g.: /path/2/file/art_painting/dog/pic_376.jpg 1
        """
        def _get_complete_domain(self, na_domain, dict_domain2pathfilepath):
            """_get_complete_domain.

            :param na_domain:
            """
            if na_domain not in self.list_domain_tr:
                trans = img_trans_te
            else:
                if self._dict_domain_img_trans:
                    trans = self._dict_domain_img_trans[na_domain]
                else:
                    trans = img_trans_tr
            root_img = self.dict_domain2imgroot[na_domain]
            path2filelist = dict_domain2pathfilepath[na_domain]
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
                # the user is required to input tr, val, te file path
                # if split=False, then only te is used, which contains
                # the whole dataset
                dset = self._get_complete_domain(
                    na_domain,
                    self._dict_domain2filepath_list_im_te)
                # test set contains train+validation
                return dset, dset  # @FIXME: avoid returning two identical

            dset = self._get_complete_domain(
                na_domain,
                # read training set from user configuration
                self._dict_domain2filepath_list_im_tr)

            dset_val = self._get_complete_domain(
                na_domain,
                # read validation set from user configuration
                self._dict_domain2filepath_list_im_val)

            return dset, dset_val

        def conf(self):
            """
            set task attribute in initialization
            """
            self.list_str_y = list_str_y
            self.isize = isize
            self.dict_class_label_ind2name = dict_class_label_ind2name
            self.dict_domain2imgroot = dict_domain2imgroot
            self._dict_domain2filepath_list_im_tr = dict_d2filepath_list_img_tr
            self._dict_domain2filepath_list_im_val = dict_d2filepath_list_img_val
            self._dict_domain2filepath_list_im_te = dict_d2filepath_list_img_te
            self.set_list_domains(list(self.dict_domain2imgroot.keys()))

        def __init__(self, succ=None):
            super().__init__(succ)
            self.conf()

    return NodeTaskPathList(succ)
