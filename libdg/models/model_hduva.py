"""
Hierarchical Domain Unsupervised Variational Auto-Encoding
"""
import torch
from torch.nn import functional as F
from torch.distributions import Dirichlet
from libdg.utils.utils_class import store_args
from libdg.models.model_vae_xyd_classif import VAEXYDClassif


class ModelHDUVA(VAEXYDClassif):
    """
    Hierarchical Domain Unsupervised Variational Auto-Encoding
    """
    @store_args
    def __init__(self, chain_node_builder,
                 zy_dim, zd_dim,
                 list_str_y, list_d_tr,
                 gamma_d, gamma_y,
                 beta_d, beta_x, beta_y,
                 beta_t,
                 device,
                 zx_dim=0,
                 topic_dim=3):
        """
        """
        super().__init__(chain_node_builder,
                         zd_dim, zy_dim, zx_dim,
                         list_str_y, list_d_tr)

        # topic to zd follows Gaussian distribution
        self.add_module("net_p_zd",
                        self.chain_node_builder.construct_cond_prior(self.topic_dim, self.zd_dim))

    def _init_components(self):
        """
        q(z|x)
        p(zy)
        q_{classif}(zy)
        """
        self.add_module("encoder", self.chain_node_builder.build_encoder(self.device, self.topic_dim))
        self.add_module("decoder", self.chain_node_builder.build_decoder(self.topic_dim))
        self.add_module("net_p_zy",
                        self.chain_node_builder.construct_cond_prior(self.dim_y, self.zy_dim))
        self.add_module("net_classif_y",
                        self.chain_node_builder.construct_classifier(self.zy_dim, self.dim_y))

    def cal_logit_y(self, tensor_x):
        """
        calculate the logit for softmax classification
        """
        q_topic, topic_q, \
            qzd, zd_q, \
            qzx, zx_q, \
            qzy, zy_q = self.encoder(tensor_x)
        logit_y = self.net_classif_y(zy_q)
        return logit_y


    def init_p_topic_batch(self, batch_size, device):
        """
        flat prior
        """
        prior = Dirichlet(torch.ones(batch_size, self.topic_dim).to(device))
        return prior

    def forward(self, x, y, d=None):
        return self.cal_loss(x, y, d)

    def cal_loss(self, x, y, d=None):
        """
        :param x:
        :param y:
        """
        q_topic, topic_q, \
            qzd, zd_q, \
            qzx, zx_q, \
            qzy, zy_q = self.encoder(x)

        batch_size = zd_q.shape[0]
        device = zd_q.device

        p_topic = self.init_p_topic_batch(batch_size, device)

        # zx KL divergence
        zx_p_minus_q = 0
        if self.zx_dim > 0:
            p_zx = self.init_p_zx4batch(batch_size, device)
            zx_p_minus_q = torch.sum(p_zx.log_prob(zx_q) - qzx.log_prob(zx_q), 1)

        # FIXME: does monte-carlo KL makes the performance unstable?
        # from torch.distributions import kl_divergence

        # zy KL divergence
        p_zy = self.net_p_zy(y)
        zy_p_minus_zy_q = torch.sum(p_zy.log_prob(zy_q) - qzy.log_prob(zy_q), 1)

        # classification loss
        logit_y = self.net_classif_y(zy_q)
        _, y_target = y.max(dim=1)
        lc_y = F.cross_entropy(logit_y, y_target, reduction="none")

        # zd KL diverence
        p_zd = self.net_p_zd(topic_q)
        zd_p_minus_q = torch.sum(p_zd.log_prob(zd_q) - qzd.log_prob(zd_q), 1)

        # topic KL divergence
        # FIXME: why topic is still there?
        topic_p_minus_q = p_topic.log_prob(topic_q) - q_topic.log_prob(topic_q)

        # reconstruction
        z_concat = self.decoder.concat_ytdx(zy_q, topic_q, zd_q, zx_q)
        loss_recon_x, _, _ = self.decoder(z_concat, x)
        batch_loss = loss_recon_x \
                     - self.beta_x * zx_p_minus_q \
                     - self.beta_y * zy_p_minus_zy_q \
                     - self.beta_d * zd_p_minus_q \
                     - self.beta_t * topic_p_minus_q \
                     + self.gamma_y * lc_y
        return batch_loss
