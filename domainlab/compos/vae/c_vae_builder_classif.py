"""
Builder
1. classifier for domain **and** class
2. p(z_y|y) **and** p(z_d|d)
"""
from domainlab.compos.nn_zoo.net_classif import ClassifDropoutReluLinear
from domainlab.compos.vae.a_vae_builder import AbstractVAEBuilderChainNode
from domainlab.compos.vae.compos.decoder_cond_prior import \
    LSCondPriorLinearBnReluLinearSoftPlus


class ChainNodeVAEBuilderClassifCondPrior(AbstractVAEBuilderChainNode):
    """
    1. This class defines common methods shared by child classes:
        - classifier for domain/class
        - conditional prior
    2. Bridge pattern: separate abstraction (vae model) and implementation)
    """
    def construct_classifier(self, input_dim, output_dim):
        """
        classifier can be used to both classify class-label and domain-label
        @param input_dim: can be both zy_dim or zd_dim
        """
        return ClassifDropoutReluLinear(input_dim, output_dim)

    def construct_cond_prior(self, input_dim, output_dim):
        """
        For both p(z_y|y) and p(z_d|d)
        """
        net_p_z_pars = LSCondPriorLinearBnReluLinearSoftPlus(input_dim, output_dim)
        return net_p_z_pars

    def build_encoder(self):
        raise NotImplementedError

    def build_decoder(self):
        raise NotImplementedError

    def is_myjob(self, request):
        return NotImplementedError
