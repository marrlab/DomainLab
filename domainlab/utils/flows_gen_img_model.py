import os

import torch

from domainlab.compos.vae.c_vae_recon import ReconVAEXYD
from domainlab.utils.utils_img_sav import mk_fun_sav_img


class FlowGenImgs():
    def __init__(self, model, device):
        model = model.to(device)
        self.obj_recon = ReconVAEXYD(model)
        self.sav_fun = None

    def _save_list(self, recon_list_d, device, img, name):
        """
        compare counterfactual
        """
        recon_stack_d = torch.cat(recon_list_d)
        recon_stack_d = recon_stack_d.to(device)
        comparison = torch.cat([img, recon_stack_d])
        self.sav_fun(comparison, name, title=None)  # row is actually column here

    def _save_pair(self, x_recon_img, device, img, name):
        """
        compare recon and original
        """
        x_recon_img = x_recon_img.to(device)
        comparison = torch.cat([img, x_recon_img])
        self.sav_fun(comparison, name)

    def gen_img_loader(self, loader, device, path, domain):
        """
        gen image for the first batch of input loader
        """
        for _, (x_batch, y_batch, *d_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            if d_batch and isinstance(d_batch[0], torch.Tensor):
                d_batch = d_batch[0].to(device)
            else:
                d_batch = None
            self.gen_img_xyd(x_batch, y_batch, d_batch, device, path, folder_na=domain)
            break

    def gen_img_xyd(self, img, vec_y, vec_d, device, path, folder_na):
        img = img.to(device)
        vec_y = vec_y.to(device)
        if vec_d is not None:
            vec_d = vec_d.to(device)
        nrow = img.shape[0]  # nrow is actually ncol in pytorch save_image!!
        self.sav_fun = mk_fun_sav_img(path=path, nrow=nrow, folder_na=folder_na)
        self._flow_vanilla(img, vec_y, vec_d, device)
        self._flow_cf_y(img, vec_y, vec_d, device)
        if vec_d is not None:
            self._flow_cf_d(img, vec_y, vec_d, device)

    def _flow_vanilla(self, img, vec_y, vec_d, device, num_sample=10):
        x_recon_img, str_type = self.obj_recon.recon(img)
        self._save_pair(x_recon_img, device, img, str_type + '.png')

        x_recon_img, str_type = self.obj_recon.recon(img, vec_y)
        self._save_pair(x_recon_img, device, img, str_type + '.png')

        if vec_d is not None:
            x_recon_img, str_type = self.obj_recon.recon(img, None, vec_d)
            self._save_pair(x_recon_img, device, img, str_type + '.png')

        for i in range(num_sample):
            x_recon_img, str_type = self.obj_recon.recon(img, vec_y, vec_d, True, True)
            x_recon_img = x_recon_img.to(device)
            comparison = torch.cat([img, x_recon_img])
            self.sav_fun(comparison, str_type + str(i) + '.png')

    def _flow_cf_y(self, img, vec_y, vec_d, device):
        """
        scan possible values of vec_y
        """
        recon_list, str_type = self.obj_recon.recon_cf(img, "y", vec_y.shape[1], device,
                                                       zx2fill=None)
        self._save_list(recon_list, device, img, "_".join(["recon_cf_y", str_type]) + ".png")
        recon_list, str_type = self.obj_recon.recon_cf(img, "y", vec_y.shape[1], device, zx2fill=0)
        self._save_list(recon_list, device, img, "_".join(["recon_cf_y", str_type]) + ".png")
        if vec_d is not None:
            recon_list, str_type = self.obj_recon.recon_cf(
                img, "y", vec_y.shape[1], device,
                vec_d=vec_d,
                zx2fill=0)
            self._save_list(recon_list, device, img, "_".join(["recon_cf_y", str_type]) + ".png")

    def _flow_cf_d(self, img, vec_y, vec_d, device):
        """
        scan possible values of vec_y
        """
        recon_list, str_type = self.obj_recon.recon_cf(img, "d", vec_d.shape[1], device,
                                                       zx2fill=None)
        self._save_list(recon_list, device, img, "_".join(["recon_cf_d", str_type]) +".png")
        recon_list, str_type = self.obj_recon.recon_cf(img, "d", vec_d.shape[1], device, zx2fill=0)
        self._save_list(recon_list, device, img, "_".join(["recon_cf_d", str_type]) +".png")


def fun_gen(model, device, node, args, subfolder_na, output_folder_na="gen"):
    flow = FlowGenImgs(model, device)
    path = os.path.join(args.out, output_folder_na, node.task_name, args.aname, subfolder_na)
    flow.gen_img_loader(node.loader_te, device,
                        path=path,
                        domain="_".join(args.te_d))
    flow.gen_img_loader(node.loader_tr, device,
                        path=path,
                        domain="_".join(node.list_domain_tr))
