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
    Instantiate a Hierarchical Domain Unsupervised VAE (HDUVA) with arbitrary task loss.

    Details:
        The created model builds on a generative approach within the framework of variational
        autoencoders to facilitate generalization to new domains without supervision. HDUVA learns
        representations that disentangle domain-specific information from class-label specific
        information even in complex settings where domain structure is not observed during training.
        Here for, latent variables are introduced, representing the information for the classes,
        domains and the residual variance of the inputs, respectively. The domain structure is
        modelled by a hierarchical level and another latent variable, denoted as topic.
        Two encoder networks are trained. One for converting an image to be compatible with the
        latent spaces of the domains and another one for converting an image to a topic
        distribution. The overall objective is constructed by adding an additional weighted term to
        the ELBO loss. One benefit of this model is that the domain information during training can
        be incomplete.
        For more details, see: Sun, Xudong, and Buettner, Florian.
        "Hierarchical variational auto-encoding for unsupervised domain generalization."
        arXiv preprint arXiv:2101.09436 (2021).

    Args:
        parent_class: Class object determining the task type. Defaults to VAEXYDClassif.

    Returns:
        ModelHDUVA: model inheriting from parent class.

    Input Parameters:
        zd_dim: size of latent space for domain-specific information (int),
        zy_dim: size of latent space for class-specific information (int),
        zx_dim: size of latent space for residual variance (int, defaults to 0),
        chain_node_builder: an object which can build different maps via neural network,
        list_str_y: list of labels (list of strings),
        gamma_d: weighting term for domain classificaiton loss
        gamma_y: weighting term for additional term in ELBO loss (float),
        beta_d: weighting term for the domain component of ELBO loss (float),
        beta_x: weighting term for residual variation component of ELBO loss (float),
        beta_y: weighting term for class component of ELBO loss (float),
        beta_t: weighting term for the topic component of ELBO loss (float),
        device: device to which the model should be moved (cpu or gpu),
        topic_dim: size of latent space for topics (int, defaults to 3)
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
            dict_rst = fun_scheduler(epoch)  # the __call__ function of hyper-para-scheduler object
            self.beta_d = dict_rst["beta_d"]
            self.beta_y = dict_rst["beta_y"]
            self.beta_x = dict_rst["beta_x"]
            self.beta_t = dict_rst["beta_t"]

        def hyper_init(self, functor_scheduler):
            """hyper_init.
            :param functor_scheduler:
            """
            # calling the constructor of the hyper-parameter-scheduler class, so that this scheduler
            # class build a dictionary {"beta_d":self.beta_d, "beta_y":self.beta_y}
            # constructor signature is def __init__(self, **kwargs):
            return functor_scheduler(
                trainer=None,
                beta_d=self.beta_d, beta_y=self.beta_y, beta_x=self.beta_x,
                beta_t=self.beta_t)

        @store_args
        def __init__(self, chain_node_builder,
                     zy_dim, zd_dim,
                     list_str_y,
                     gamma_d, gamma_y,
                     beta_d, beta_x, beta_y,
                     beta_t,
                     device,
                     zx_dim=0,
                     topic_dim=3,
                     multiplier_recon=1.0):
            """
            """
            super().__init__(chain_node_builder,
                             zd_dim, zy_dim, zx_dim,
                             list_str_y)

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

        def _cal_reg_loss(self, tensor_x, tensor_y, tensor_d=None, others=None):
            q_topic, topic_q, \
                qzd, zd_q, \
                qzx, zx_q, \
                qzy, zy_q = self.encoder(tensor_x)

            batch_size = zd_q.shape[0]
            device = zd_q.device

            p_topic = self.init_p_topic_batch(batch_size, device)

            # @FIXME: does monte-carlo KL makes the performance unstable?
            # from torch.distributions import kl_divergence

            # zy KL divergence

            if (tensor_y.shape[-1] == 1) | (len(tensor_y.shape) == 1):
                tensor_y_onehot = torch.nn.functional.one_hot(
                    tensor_y,
                    num_classes=len(self.list_str_y))
                tensor_y_onehot = tensor_y_onehot.to(torch.float32)
            else:
                tensor_y_onehot = tensor_y

            p_zy = self.net_p_zy(tensor_y_onehot)
            zy_p_minus_zy_q = g_inst_component_loss_agg(
                p_zy.log_prob(zy_q) - qzy.log_prob(zy_q), 1)

            # zx KL divergence
            zx_p_minus_q = torch.zeros_like(zy_p_minus_zy_q)
            if self.zx_dim > 0:
                p_zx = self.init_p_zx4batch(batch_size, device)
                zx_p_minus_q = g_inst_component_loss_agg(
                    p_zx.log_prob(zx_q) - qzx.log_prob(zx_q), 1)

            # zd KL diverence
            p_zd = self.net_p_zd(topic_q)
            zd_p_minus_q = g_inst_component_loss_agg(p_zd.log_prob(zd_q) - qzd.log_prob(zd_q), 1)

            # topic KL divergence
            # @FIXME: why topic is still there?
            topic_p_minus_q = p_topic.log_prob(topic_q) - q_topic.log_prob(topic_q)

            # reconstruction
            z_concat = self.decoder.concat_ytdx(zy_q, topic_q, zd_q, zx_q)
            loss_recon_x, _, _ = self.decoder(z_concat, tensor_x)
            return [loss_recon_x, zx_p_minus_q, zy_p_minus_zy_q, zd_p_minus_q, topic_p_minus_q], \
                [self.multiplier_recon, -self.beta_x, -self.beta_y, -self.beta_d, -self.beta_t]

        def extract_semantic_feat(self, tensor_x):
            """
            :param tensor_x:
            """
            zy_q_loc = self.encoder.infer_zy_loc(tensor_x)
            return zy_q_loc

    return ModelHDUVA
