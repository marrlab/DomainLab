"""
DIVA
"""
import torch
from torch.nn import functional as F

from libdg.models.model_vae_xyd_classif import VAEXYDClassif
from libdg.utils.utils_class import store_args


class ModelDIVA(VAEXYDClassif):
    """
    DIVA
    """
    @store_args
    def __init__(self, chain_node_builder,
                 zd_dim, zy_dim, zx_dim,
                 list_str_y, list_d_tr,
                 gamma_d, gamma_y,
                 beta_d, beta_x, beta_y):
        """
        gamma: classification loss coefficient
        """
        super().__init__(chain_node_builder,
                         zd_dim, zy_dim, zx_dim,
                         list_str_y, list_d_tr)
        self.dim_d_tr = len(self.list_d_tr)
        if self.zd_dim > 0:
            self.add_module(
                "net_p_zd",
                self.chain_node_builder.construct_cond_prior(
                    self.dim_d_tr, self.zd_dim))
            self.add_module(
                "net_classif_d",
                self.chain_node_builder.construct_classifier(
                    self.zd_dim, self.dim_d_tr))

    def warm_up_beta(self, epoch, beta_steady=1.0, steps=100):
        self.beta_d = min([beta_steady,
                           beta_steady * ((epoch+1) * 1.) / steps])  # for zd
        self.beta_y = min([beta_steady,
                           beta_steady * ((epoch+1) * 1.) / steps])  # for zy
        self.beta_x = min([beta_steady,
                           beta_steady * ((epoch+1) * 1.) / steps])  # for zx

    def get_list_str_y(self):
        return self._list_str_y

    def forward(self, x, y, d):
        q_zd, zd_q, q_zx, zx_q, q_zy, zy_q = self.encoder(x)
        logit_d = self.net_classif_d(zd_q)
        logit_y = self.net_classif_y(zy_q)

        batch_size = zd_q.shape[0]
        device = zd_q.device

        p_zx = self.init_p_zx4batch(batch_size, device)
        p_zy = self.net_p_zy(y)
        p_zd = self.net_p_zd(d)

        z_concat = self.decoder.concat_ydx(zy_q, zd_q, zx_q)
        loss_recon_x, _, _ = self.decoder(z_concat, x)

        zd_p_minus_zd_q = torch.sum(
            p_zd.log_prob(zd_q) - q_zd.log_prob(zd_q), 1)
        zx_p_minus_zx_q = torch.sum(
            p_zx.log_prob(zx_q) - q_zx.log_prob(zx_q), 1)
        zy_p_minus_zy_q = torch.sum(
            p_zy.log_prob(zy_q) - q_zy.log_prob(zy_q), 1)

        _, d_target = d.max(dim=1)
        lc_d = F.cross_entropy(logit_d, d_target, reduction="none")
        _, y_target = y.max(dim=1)
        lc_y = F.cross_entropy(logit_y, y_target, reduction="none")
        return loss_recon_x \
            - self.beta_d * zd_p_minus_zd_q \
            - self.beta_x * zx_p_minus_zx_q \
            - self.beta_y * zy_p_minus_zy_q \
            + self.gamma_d * lc_d \
            + self.gamma_y * lc_y, \
            loss_recon_x,  \
            zd_p_minus_zd_q, \
            zx_p_minus_zx_q, \
            zy_p_minus_zy_q, \
            lc_y, \
            lc_d

def test_fun():
    from libdg.compos.vae.utils_request_chain_builder import RequestVAEBuilderCHW, VAEChainNodeGetter
    from libdg.utils.test_img import mk_rand_xyd
    from libdg.utils.utils_classif import mk_dummy_label_list_str

    im_h = 64
    y_dim = 10
    dim_d_tr = 3
    batch_size = 5

    request = RequestVAEBuilderCHW(3, im_h, im_h)
    node = VAEChainNodeGetter(request)()

    list_str_y = mk_dummy_label_list_str("class", y_dim)
    list_d_tr = mk_dummy_label_list_str("domain", dim_d_tr)

    model = ModelDIVA(node, zd_dim=8, zy_dim=8, zx_dim=8, gamma_d=1.0, gamma_y=1.0,
                      list_str_y=list_str_y, list_d_tr=list_d_tr)
    imgs, y_s, d_s = mk_rand_xyd(im_h, y_dim, dim_d_tr, batch_size)
    one_hot, mat_prob, label, confidence, na = model.infer_y_vpicn(imgs)
    model(imgs, y_s, d_s)
    return model, imgs, y_s, d_s
