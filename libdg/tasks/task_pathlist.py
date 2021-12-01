"""
When class names and numbers does not match across different domains
"""
import os
from torch.utils.data.dataset import ConcatDataset
from torchvision import transforms
from libdg.tasks.b_task import NodeTaskDict
from libdg.dsets.utils_data import mk_fun_label2onehot, fun_img_path_loader_default
from libdg.dsets.dset_img_path_list import DsetImPathList
from libdg.dsets.utils_data import mk_fun_label2onehot



def mk_node_task_path_list(isize,
                           list_str_y,
                           trans4all,
                           dict_class_label2name,
                           dict_domain2imgroot,
                           dict_d2filepath_list_img,
                           succ=None):
    class NodeTaskPathList(NodeTaskDict):
        """
        imgpath: art_painting/dog/pic_376.jpg 1
        """
        def get_dset_by_domain(self, args, na_domain):
            if self._dict_domain_img_trans:
                trans = self._dict_domain_img_trans[na_domain]
            else:
                trans = transforms.ToTensor()
            root_img = self.dict_domain2imgroot[na_domain]
            path2filelist = self._dict_domain2filepath_list_im[na_domain]
            path2filelist = os.path.expanduser(path2filelist)
            root_img = os.path.expanduser(root_img)
            dset = DsetImPathList(root_img, path2filelist, trans_img=trans,
                                  trans_target=mk_fun_label2onehot(len(self.list_str_y)))
            return dset

        def conf(self, args):
            self.list_str_y = list_str_y
            self.isize = isize
            self.dict_domain2imgroot = dict_domain2imgroot
            self._dict_domain2filepath_list_im = dict_d2filepath_list_img
            self.set_list_domains(list(self.dict_domain2imgroot.keys()))

        def init_business(self, args):
            self.conf(args)
            super().init_business(args)

    return NodeTaskPathList(succ)



def test_fun():
    """
    """
    from libdg.utils.arg_parser import mk_parser_main
    from libdg.tasks.utils_task import img_loader2dir, ImSize


    node = mk_node_task_path_list(
        isize=ImSize(3, 224, 224),
        list_str_y=["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"],
        dict_class_label2name={"1": "dog",
                               "2": "elephant",
                               "3": "giraffe",
                               "4": "guitar",
                               "5": "horse",
                               "6": "house",
                               "7": "person"},
        dict_d2filepath_list_img={
            "art_painting": "~/Documents/dataset/PACS/h5kfold/art_painting_train_kfold.txt",
            "cartoon": "~/Documents/dataset/PACS/h5kfold/cartoon_train_kfold.txt",
            "photo": "~/Documents/dataset/PACS/h5kfold/photo_train_kfold.txt",
            "sketch": "~/Documents/dataset/PACS/h5kfold/sketch_train_kfold.txt"},
        dict_domain2imgroot={
            'art_painting': "~/Documents/dataset/PACS/raw",
            'cartoon': "~/Documents/dataset/PACS/raw",
            'photo': "~/Documents/dataset/PACS/raw",
            'sketch': "~/Documents/dataset/PACS/raw"},
        trans4all=transforms.ToTensor())


    parser = mk_parser_main()
    args = parser.parse_args(["--te_d", "1"])
    node.init_business(args)


    img_loader2dir(node._loader_te, list_domain_na=node.get_list_domains(),
                   list_class_na=node.list_str_y, folder="zout/test_loader/pacs", batches=10)
    img_loader2dir(node._loader_tr, list_domain_na=node.get_list_domains(),
                   list_class_na=node.list_str_y, folder="zout/test_loader/pacs", batches=10)
