"""
Chain node VAE builders
"""
from domainlab.compos.vae.compos.decoder_concat_vec_reshape_conv_gated_conv import \
    DecoderConcatLatentFCReshapeConvGatedConv
from domainlab.compos.vae.compos.encoder_xydt_elevator import XYDTEncoderArg
from domainlab.compos.vae.zoo_vae_builders_classif import (NodeVAEBuilderArg, NodeVAEBuilderUser)


class NodeVAEBuilderImgTopic(NodeVAEBuilderArg):
    """NodeVAEBuilderImgTopic."""
    def is_myjob(self, request):
        """is_myjob.

        :param request:
        """
        self.args = request.args
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
                                 topic_h_dim=self.args.topic_h_dim,
                                 img_h_dim=self.args.img_h_dim,
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
    """NodeVAEBuilderImgTopic if user input does not come via command line, but from test file"""


    def is_myjob(self, request):
        flag = not hasattr(request, "args")
        self.config_img(flag, request)
        return flag


    def build_encoder(self):  # TODO
        # encoder = XYDTEncoderUser(
        return


    def build_decoder(self):  # TODO
        return