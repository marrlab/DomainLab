"""
Chain node VAE builders
"""
from libdg.compos.vae.c_vae_builder_classif import \
    ChainNodeVAEBuilderClassifCondPrior
from libdg.compos.vae.compos.encoder_xyd_parallel import \
    XYDEncoderParallelConvBnReluPool, XYDEncoderParallelAlex, \
    XYDEncoderParallelExtern
from libdg.compos.vae.compos.decoder_concat_vec_reshape_conv_gated_conv \
    import DecoderConcatLatentFCReshapeConvGatedConv
from libdg.compos.vae.compos.encoder_xydt_elevator import XYDTEncoderAlex
from libdg.compos.vae.compos.encoder_xydt_elevator import \
    XYDTEncoderConvBnReluPool


class ChainNodeVAEBuilderClassifCondPriorBase(
        ChainNodeVAEBuilderClassifCondPrior):
    """
    base class of AE builder
    """
    def config_img(self, flag, request):
        """config_img.

        :param flag:
        :param request:
        """
        if flag:
            self.i_c = request.i_c
            self.i_h = request.i_h
            self.i_w = request.i_w

    def is_myjob(self, request):
        """is_myjob.

        :param request:
        """
        raise NotImplementedError

    def build_encoder(self):
        """build_encoder."""
        raise NotImplementedError

    def build_decoder(self):
        """build_decoder."""
        decoder = DecoderConcatLatentFCReshapeConvGatedConv(
            z_dim=self.zd_dim+self.zx_dim+self.zy_dim,
            i_c=self.i_c, i_w=self.i_w,
            i_h=self.i_h)
        return decoder


class NodeVAEBuilderArg(ChainNodeVAEBuilderClassifCondPriorBase):
    """Build encoder decoder according to commandline arguments
    """
    def is_myjob(self, request):
        """is_myjob.
        :param request:
        """
        self.args = request.args
        if self.args.npath is not None:
            return True
        return False

    def build_encoder(self):
        """build_encoder."""
        encoder = XYDEncoderParallelExtern(
            self.zd_dim, self.zx_dim, self.zy_dim, args=self.args)
        return encoder


class NodeVAEBuilderImg28(ChainNodeVAEBuilderClassifCondPriorBase):
    def is_myjob(self, request):
        """is_myjob.

        :param request:
        """
        flag = (request.i_h == 28)
        self.config_img(flag, request)
        return flag

    def build_encoder(self):
        """build_encoder."""
        encoder = XYDEncoderParallelConvBnReluPool(
            self.zd_dim, self.zx_dim, self.zy_dim,
            self.i_c,
            self.i_h,
            self.i_w)
        return encoder


class NodeVAEBuilderImg64(NodeVAEBuilderImg28):
    """NodeVAEBuilderImg64."""

    def is_myjob(self, request):
        """is_myjob.

        :param request:
        """
        flag = (request.i_h == 64)
        self.config_img(flag, request)
        return flag


class NodeVAEBuilderImg224(NodeVAEBuilderImg28):
    """NodeVAEBuilderImg224."""

    def is_myjob(self, request):
        """is_myjob.

        :param request:
        """
        self.args = request.args
        flag = (request.i_h == 224)
        self.config_img(flag, request)
        return flag

    def build_encoder(self):
        """build_encoder."""
        encoder = XYDEncoderParallelAlex(
            self.zd_dim, self.zx_dim, self.zy_dim,
            self.i_c,
            self.i_h,
            self.i_w, args=self.args)
        return encoder


class NodeVAEBuilderImgTopic(NodeVAEBuilderArg):
    """NodeVAEBuilderImgTopic."""
    def is_myjob(self, request):
        """is_myjob.

        :param request:
        """
        self.args = request.args
        if self.args.npath is not None:
            return True
        flag = (request.i_h == 224)
        self.config_img(flag, request)
        return flag

    def build_encoder(self, device, topic_dim):
        """build_encoder.

        :param device:
        :param topic_dim:
        """
        encoder = XYDTEncoderAlex(device, topic_dim,
                                  self.zd_dim, self.zx_dim,
                                  self.zy_dim,
                                  self.i_c,
                                  self.i_h,
                                  self.i_w, conv_stride=1,
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


class NodeVAEBuilderImgTopicMNIST(NodeVAEBuilderImgTopic):
    def is_myjob(self, request):
        """is_myjob.

        :param request:
        """
        self.args = request.args
        if self.args.npath is not None:
            return False
        flag = (request.i_h < 100)  # FIXME: should be decided by user
        self.config_img(flag, request)
        return flag

    def build_encoder(self, device, topic_dim):
        """build_encoder.

        :param device:
        :param topic_dim:
        """
        encoder = XYDTEncoderConvBnReluPool(
            device, topic_dim,
            self.zd_dim, self.zx_dim,
            self.zy_dim,
            self.i_c,
            self.i_h,
            self.i_w, conv_stride=1)
        return encoder
