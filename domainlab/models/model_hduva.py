"""
Hierarchical Domain Unsupervised Variational Auto-Encoding
"""
import torch
from torch.distributions import Dirichlet

from domainlab.models.model_vae_xyd_classif import VAEXYDClassif
from domainlab.utils.utils_class import store_args
from domainlab import g_inst_component_loss_agg


def mk_hduva(parent_class=VAEXYDClassif):
    """
    Hierarchical Domain Unsupervised VAE with arbitrary task loss
    """
    class ModelHDUVA(parent_class):
        """
        Hierarchical Domain Unsupervised Variational Auto-Encoding
        """
        def hyper_update(self, epoch, fun_scheduler):
            """hyper_update.

            :param epoch:
            :param fun_scheduler:
            """
            dict_rst = fun_scheduler(epoch)
            self.beta_d = dict_rst["beta_d"]
            self.beta_y = dict_rst["beta_y"]
            self.beta_x = dict_rst["beta_x"]
            self.beta_t = dict_rst["beta_t"]

        def hyper_init(self, functor_scheduler):
            """hyper_init.
            :param functor_scheduler:
            """
            return functor_scheduler(
                beta_d=self.beta_d, beta_y=self.beta_y, beta_x=self.beta_x,
                beta_t=self.beta_t)

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
                            self.chain_node_builder.construct_cond_prior(
                                self.topic_dim, self.zd_dim))

        def _init_components(self):
            """
            q(z|x)
            p(zy)
            q_{classif}(zy)
            """
            self.add_module("encoder", self.chain_node_builder.build_encoder(
                self.device, self.topic_dim))
            self.add_module("decoder", self.chain_node_builder.build_decoder(
                self.topic_dim))
            self.add_module("net_p_zy",
                            self.chain_node_builder.construct_cond_prior(
                                self.dim_y, self.zy_dim))
            self.add_module("net_classif_y",
                            self.chain_node_builder.construct_classifier(
                                self.zy_dim, self.dim_y))

        def init_p_topic_batch(self, batch_size, device):
            """
            flat prior
            """
            prior = Dirichlet(torch.ones(batch_size, self.topic_dim).to(device))
            return prior

        def cal_reg_loss(self, tensor_x, tensor_y, tensor_d=None, others=None):
            q_topic, topic_q, \
                qzd, zd_q, \
                qzx, zx_q, \
                qzy, zy_q = self.encoder(tensor_x)

            batch_size = zd_q.shape[0]
            device = zd_q.device

            p_topic = self.init_p_topic_batch(batch_size, device)

            # zx KL divergence
            zx_p_minus_q = 0
            if self.zx_dim > 0:
                p_zx = self.init_p_zx4batch(batch_size, device)
                zx_p_minus_q = g_inst_component_loss_agg(
                    p_zx.log_prob(zx_q) - qzx.log_prob(zx_q), 1)

            # @FIXME: does monte-carlo KL makes the performance unstable?
            # from torch.distributions import kl_divergence

            # zy KL divergence
            p_zy = self.net_p_zy(tensor_y)
            zy_p_minus_zy_q = g_inst_component_loss_agg(p_zy.log_prob(zy_q) - qzy.log_prob(zy_q), 1)

            # zd KL diverence
            p_zd = self.net_p_zd(topic_q)
            zd_p_minus_q = g_inst_component_loss_agg(p_zd.log_prob(zd_q) - qzd.log_prob(zd_q), 1)

            # topic KL divergence
            # @FIXME: why topic is still there?
            topic_p_minus_q = p_topic.log_prob(topic_q) - q_topic.log_prob(topic_q)

            # reconstruction
            z_concat = self.decoder.concat_ytdx(zy_q, topic_q, zd_q, zx_q)
            loss_recon_x, _, _ = self.decoder(z_concat, tensor_x)
            batch_loss = loss_recon_x \
                - self.beta_x * zx_p_minus_q \
                - self.beta_y * zy_p_minus_zy_q \
                - self.beta_d * zd_p_minus_q \
                - self.beta_t * topic_p_minus_q
            return batch_loss

        def extract_semantic_features(self, tensor_x):
            """
            :param tensor_x:
            """
            zy_q_loc = self.encoder.infer_zy_loc(tensor_x)
            return zy_q_loc

    return ModelHDUVA
