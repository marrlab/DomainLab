"""
DIVA
"""
from torch.nn import functional as F

from domainlab import g_inst_component_loss_agg
from domainlab.models.model_vae_xyd_classif import VAEXYDClassif
from domainlab.utils.utils_class import store_args


def mk_diva(parent_class=VAEXYDClassif):
    """
    DIVA with arbitrary task loss
    """
    class ModelDIVA(parent_class):
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

        def cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
            q_zd, zd_q, q_zx, zx_q, q_zy, zy_q = self.encoder(tensor_x)
            logit_d = self.net_classif_d(zd_q)

            batch_size = zd_q.shape[0]
            device = zd_q.device

            p_zx = self.init_p_zx4batch(batch_size, device)
            p_zy = self.net_p_zy(tensor_y)
            p_zd = self.net_p_zd(tensor_d)

            z_concat = self.decoder.concat_ydx(zy_q, zd_q, zx_q)
            loss_recon_x, _, _ = self.decoder(z_concat, tensor_x)

            zd_p_minus_zd_q = g_inst_component_loss_agg(
                p_zd.log_prob(zd_q) - q_zd.log_prob(zd_q), 1)
            # without aggregation, shape is [batchsize, zd_dim]
            zx_p_minus_zx_q = 0
            if self.zx_dim > 0:
                # torch.sum will return 0 for empty tensor,
                # torch.mean will return nan
                zx_p_minus_zx_q = g_inst_component_loss_agg(
                    p_zx.log_prob(zx_q) - q_zx.log_prob(zx_q), 1)

            zy_p_minus_zy_q = g_inst_component_loss_agg(
                p_zy.log_prob(zy_q) - q_zy.log_prob(zy_q), 1)

            _, d_target = tensor_d.max(dim=1)
            lc_d = F.cross_entropy(logit_d, d_target, reduction="none")

            loss_reg = loss_recon_x \
                - self.beta_d * zd_p_minus_zd_q \
                - self.beta_x * zx_p_minus_zx_q \
                - self.beta_y * zy_p_minus_zy_q \
                + self.gamma_d * lc_d
            return loss_reg
    return ModelDIVA
