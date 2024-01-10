"""
Chain node VAE builders
"""
from domainlab.compos.vae.compos.decoder_concat_vec_reshape_conv_gated_conv import \
    DecoderConcatLatentFCReshapeConvGatedConv
from domainlab.compos.vae.compos.encoder_xydt_elevator import (XYDTEncoderArg, XYDTEncoderArgUser)
from domainlab.compos.vae.zoo_vae_builders_classif import NodeVAEBuilderArg, NodeVAEBuilderUser


class NodeVAEBuilderImgTopic(NodeVAEBuilderArg):
    """NodeVAEBuilderImgTopic."""
    def is_myjob(self, request):
        """is_myjob.

        :param request:
        """
        flag = hasattr(request, "args")
        self.config_img(flag, request)
        return flag

    def build_encoder(self, device, topic_dim):
        """build_encoder.

        :param device:
        :param topic_dim:
        """
        encoder = XYDTEncoderArg(device, topic_dim,
                                 self.zd_dim, self.zx_dim,
                                 self.zy_dim,
                                 self.i_c,
                                 self.i_h,
                                 self.i_w,
                                 args=self.args)
        return encoder

    def build_decoder(self, topic_dim):
        """build_decoder.

        :param topic_dim:
        """
        decoder = DecoderConcatLatentFCReshapeConvGatedConv(
            z_dim=self.zd_dim+self.zx_dim+self.zy_dim+topic_dim,
            i_c=self.i_c, i_w=self.i_w,
            i_h=self.i_h)
        return decoder


class NodeVAEBuilderImgTopicUser(NodeVAEBuilderUser):
    """NodeVAEBuilderImgTopic if user input does not come from command line"""

    def is_myjob(self, request):
        """is_myjob.

        :param request:
        """
        flag = not hasattr(request, "args")
        self.request = request
        self.config_img(flag, request)
        return flag


    def build_encoder(self, device, topic_dim):
        """build_encoder.

        :param device:
        :param topic_dim:
        """
        encoder = XYDTEncoderArgUser(self.request.net_class_d,
                                     self.request.net_x,
                                     self.request.net_class_y)
        return encoder


    def build_decoder(self, topic_dim):
        """build_decoder.

        :param topic_dim:
        """
        decoder = DecoderConcatLatentFCReshapeConvGatedConv(
            z_dim=self.zd_dim+self.zx_dim+self.zy_dim+topic_dim,
            i_c=self.i_c, i_w=self.i_w,
            i_h=self.i_h)
        return decoder
