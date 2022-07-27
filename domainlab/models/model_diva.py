"""
DIVA
"""
import torch
from torch.nn import functional as F

from domainlab.models.model_vae_xyd_classif import VAEXYDClassif
from domainlab.utils.utils_class import store_args


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

    def hyper_update(self, epoch, fun_scheduler):
        """hyper_update.

        :param epoch:
        :param fun_scheduler:
        """
        dict_rst = fun_scheduler(epoch)
        self.beta_d = dict_rst["beta_d"]
        self.beta_y = dict_rst["beta_y"]
        self.beta_x = dict_rst["beta_x"]

    def hyper_init(self, functor_scheduler):
        """hyper_init.
        :param functor_scheduler:
        """
        return functor_scheduler(
            beta_d=self.beta_d, beta_y=self.beta_y, beta_x=self.beta_x)

    def get_list_str_y(self):
        """get_list_str_y."""
        return self._list_str_y

    def forward(self, x, y, d):
        """forward.

        :param x:
        :param y:
        :param d:
        """
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

    def cal_loss(self, tensor_x, tensor_y, tensor_d):
        """cal_loss.

        :param tensor_x:
        :param tensor_y:
        :param tensor_d:
        """
        loss, *_ = self.forward(tensor_x, tensor_y, tensor_d)
        return loss
