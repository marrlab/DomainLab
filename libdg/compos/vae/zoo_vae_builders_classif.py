"""
Chain node VAE builders
"""
from libdg.compos.vae.c_vae_builder_classif import ChainNodeVAEBuilderClassifCondPrior
from libdg.compos.vae.compos.encoder_xyd_parallel import XYDEncoderParallelConvBnReluPool, XYDEncoderParallelAlex
from libdg.compos.vae.compos.decoder_concat_vec_reshape_conv_gated_conv import \
    DecoderConcatLatentFCReshapeConvGatedConv
from libdg.compos.vae.compos.encoder_xydt_elevator import XYDTEncoderAlex


class NodeVAEBuilderImg28(ChainNodeVAEBuilderClassifCondPrior):
    """
    For color mnist
    """
    def config_img(self, flag, request):
        if flag:
            self.i_c = request.i_c
            self.i_h = request.i_h
            self.i_w = request.i_w

    def is_myjob(self, request):
        flag = (request.i_h == 28)
        self.config_img(flag, request)
        return flag

    def build_encoder(self):
        encoder = XYDEncoderParallelConvBnReluPool(
            self.zd_dim, self.zx_dim, self.zy_dim,
            self.i_c,
            self.i_h,
            self.i_w)
        return encoder

    def build_decoder(self):
        decoder = DecoderConcatLatentFCReshapeConvGatedConv(
            z_dim=self.zd_dim+self.zx_dim+self.zy_dim,
            i_c=self.i_c, i_w=self.i_w,
            i_h=self.i_h)
        return decoder


class NodeVAEBuilderImg64(NodeVAEBuilderImg28):
    def is_myjob(self, request):
        flag = (request.i_h == 64)
        self.config_img(flag, request)
        return flag


class NodeVAEBuilderImg224(NodeVAEBuilderImg28):
    def is_myjob(self, request):
        flag = (request.i_h == 224)
        self.config_img(flag, request)
        return flag

    def build_encoder(self):
        encoder = XYDEncoderParallelAlex(
            self.zd_dim, self.zx_dim, self.zy_dim,
            self.i_c,
            self.i_h,
            self.i_w)
        return encoder


class NodeVAEBuilderImg224Topic(NodeVAEBuilderImg224):
    def build_encoder(self, device, topic_dim):
        encoder = XYDTEncoderAlex(device, topic_dim,
                                  self.zd_dim, self.zx_dim,
                                  self.zy_dim,
                                  self.i_c,
                                  self.i_h,
                                  self.i_w, conv_stride=1)
        return encoder

    def build_decoder(self, topic_dim):
        decoder = DecoderConcatLatentFCReshapeConvGatedConv(
            z_dim=self.zd_dim+self.zx_dim+self.zy_dim+topic_dim,
            i_c=self.i_c, i_w=self.i_w,
            i_h=self.i_h)
        return decoder
