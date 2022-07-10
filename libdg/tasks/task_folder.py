"""
When class names and numbers does not match across different domains
"""
from torchvision import transforms
from libdg.tasks.b_task import NodeTaskDict
from libdg.tasks.utils_task import DsetClassVecDecoratorImgPath
from libdg.dsets.dset_subfolder import DsetSubFolder
from libdg.dsets.utils_data import mk_fun_label2onehot, \
    fun_img_path_loader_default
from libdg.dsets.utils_data import DsetInMemDecorator


class NodeTaskFolder(NodeTaskDict):
    """
    """
    @property
    def dict_domain2imgroot(self):
        """
        {"domain name":"xx/yy/zz"}
        """
        return self._dict_domains2imgroot

    @dict_domain2imgroot.setter
    def dict_domain2imgroot(self, dict_root):
        """
        {"domain name":"xx/yy/zz"}
        """
        if not isinstance(dict_root, dict):
            raise RuntimeError("input is not diciontary")
        self._dict_domains2imgroot = dict_root

    @property
    def extensions(self):
        return self.dict_att["img_extensions"]

    @extensions.setter
    def extensions(self, str_format):
        self.dict_att["img_extensions"] = str_format

    def get_dset_by_domain(self, args, na_domain, split=False):
        if float(args.split):
            raise RuntimeError(
                "this task does not support spliting training domain yet")
        if self._dict_domain_img_trans:
            trans = self._dict_domain_img_trans[na_domain]
            if na_domain not in self.list_domain_tr:
                trans = self.img_trans_te
        else:
            trans = transforms.ToTensor()
        dset = DsetSubFolder(root=self.dict_domain2imgroot[na_domain],
                             list_class_dir=self.list_str_y,
                             loader=fun_img_path_loader_default,
                             extensions=self.extensions,
                             transform=trans,
                             target_transform=mk_fun_label2onehot(len(self.list_str_y)))
        return dset, dset  # FIXME: validation by default set to be training set


class NodeTaskFolderClassNaMismatch(NodeTaskFolder):
    """
    when the folder names of the same class from different domains have
    different names
    """
    def get_dset_by_domain(self, args, na_domain, split=False):
        if float(args.split):
            raise RuntimeError(
                "this task does not support spliting training domain yet")
        print("reading domain:", na_domain)
        domain_class_dirs = \
            self._dict_domain_folder_name2class[na_domain].keys()
        if self._dict_domain_img_trans:
            trans = self._dict_domain_img_trans[na_domain]
            if na_domain not in self.list_domain_tr:
                trans = self.img_trans_te
        else:
            trans = transforms.ToTensor()
        dset = DsetSubFolder(root=self.dict_domain2imgroot[na_domain],
                             list_class_dir=list(domain_class_dirs),
                             loader=fun_img_path_loader_default,
                             extensions=self.extensions[na_domain],
                             transform=trans,
                             target_transform=mk_fun_label2onehot(
                                 len(self.list_str_y)))
        # dset.path2imgs
        dict_folder_name2class_global = \
            self._dict_domain_folder_name2class[na_domain]
        dset = DsetClassVecDecoratorImgPath(
            dset, dict_folder_name2class_global, self.list_str_y)
        # Always use the DsetInMemDecorator at the last step
        # since it does not have other needed attributes in bewteen
        if args.dmem:
            dset = DsetInMemDecorator(dset, na_domain)
        return dset, dset # FIXME: validation by default set to be training set
