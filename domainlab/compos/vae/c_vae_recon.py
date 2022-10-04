"""
Adaptor is vital for data generation so this class can be decoupled from model class.
The model class can be refactored, we do want to use the trained old-version model, which we
only need to change adaptor class.
"""
import torch

from domainlab.compos.vae.c_vae_adaptor_model_recon import AdaptorReconVAEXYD


class ReconVAEXYD():
    """
    Adaptor is vital for data generation so this class can be decoupled from model class.
    The model class can be refactored, we do want to use the trained old-version model, which we
    only need to change adaptor class.
    """
    def __init__(self, model, na_adaptor=AdaptorReconVAEXYD):
        self.model = model
        self.adaptor = na_adaptor(self.model)

    def recon(self, x, vec_y=None, vec_d=None,
              sample_p_zy=False, sample_p_zd=False,
              scalar_zx2fill=None):
        """
        common function
        """
        str_type = ""

        with torch.no_grad():
            q_zd, q_zx, q_zy = self.adaptor.cal_latent(x)

            if vec_d is None:
                recon_zd = q_zd.mean
                str_type = "zd_q"
            else:
                p_zd = self.adaptor.cal_prior_zd(vec_d)
                recon_zd = p_zd.mean
                str_type = "zd_p"
                if sample_p_zd:
                    recon_zd = p_zd.rsample()
                    str_type = "_".join([str_type, "sample__"])

            zx_loc_q = q_zx.mean
            if scalar_zx2fill is not None:
                recon_zx = torch.zeros_like(zx_loc_q)
                recon_zx = recon_zx.fill_(scalar_zx2fill)
                str_type = "_".join([str_type, "__fill_zx_", str(scalar_zx2fill), "___"])
            else:
                recon_zx = zx_loc_q
                str_type = "_".join([str_type, "__zx_q__"])

            if vec_y is not None:
                p_zy = self.adaptor.cal_prior_zy(vec_y)
                recon_zy = p_zy.mean
                str_type = "_".join([str_type, "__zy_p__"])
                if sample_p_zy:
                    recon_zy = p_zy.rsample()
                    str_type = "_".join([str_type, "sample__"])
            else:
                recon_zy = q_zy.mean
                str_type = "_".join([str_type, "zy_q__"])
            img_recon = self.adaptor.recon_ydx(recon_zy, recon_zd, recon_zx, x)
            return img_recon, str_type

    def recon_cf(self, x, na_cf, dim_cf, device,
                 vec_y=None, vec_d=None,
                 zx2fill=None,
                 sample_p_zy=False, sample_p_zd=False):
        """
        Countefactual reconstruction:
            :param na_cf: name of counterfactual, 'y' or 'd'
            :param dim_cf: dimension of counterfactual factor
        """
        list_recon_cf = []
        batch_size = x.shape[0]
        str_type = None
        # Counterfactual image generation
        for i in range(dim_cf):
            label_cf = torch.zeros(batch_size, dim_cf).to(device)
            label_cf[:, i] = 1
            if na_cf == "y":
                img_recon_cf, str_type = self.recon(x, label_cf, vec_d, sample_p_zy, sample_p_zd, zx2fill)
            elif na_cf == "d":
                img_recon_cf, str_type = self.recon(x, vec_y, label_cf, sample_p_zy, sample_p_zd, zx2fill)
            else:
                raise RuntimeError("counterfactual image generation can only be 'y' or 'd'")
            list_recon_cf.append(img_recon_cf)
        return list_recon_cf, str_type
