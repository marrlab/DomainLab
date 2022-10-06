import torch.nn as nn

from domainlab.compos.vae.compos.encoder import LSEncoderConvBnReluPool
from domainlab.compos.vae.compos.encoder_domain_topic import \
    EncoderImg2TopicDirZd
from domainlab.compos.vae.compos.encoder_zy import \
    EncoderConnectLastFeatLayer2Z
from domainlab.utils.utils_class import store_args


class XYDTEncoderElevator(nn.Module):
    """
    x->zx, x->zy, x->s, (x,s)->zd
    """
    def __init__(self, net_infer_zd_topic, net_infer_zx, net_infer_zy):
        super().__init__()
        self.add_module("net_infer_zd_topic", net_infer_zd_topic)
        self.add_module("net_infer_zx", net_infer_zx)
        self.add_module("net_infer_zy", net_infer_zy)

    def forward(self, img):
        """
        encode img into 3 latent variables separately/parallel
        :param img:
        :return q_zd, zd_q, q_zx, zx_q, q_zy, zy_q
        """
        q_topic, topic_q, q_zd, zd_q = self.net_infer_zd_topic(img)
        q_zx, zx_q = self.net_infer_zx(img)
        q_zy, zy_q = self.net_infer_zy(img)
        return q_topic, topic_q, q_zd, zd_q, q_zx, zx_q, q_zy, zy_q

    def infer_zy_loc(self, tensor):
        """
        Used by VAE model to predict class label
        """
        q_zy, _ = self.net_infer_zy(tensor)
        zy_loc = q_zy.mean
        return zy_loc


class XYDTEncoderArg(XYDTEncoderElevator):
    """
    This class only reimplemented constructor of parent class
    """
    @store_args
    def __init__(self, device, topic_dim, zd_dim,
                 zx_dim, zy_dim, i_c, i_h, i_w,
                 args,
                 topic_h_dim,
                 img_h_dim):
        """
        :param zd_dim:
        :param zx_dim:
        :param zy_dim:
        :param i_c: number of image channels
        :param i_h: image height
        :param i_w: image width
        :param img_h_dim: (img->h_img, topic->h_topic)-> q_zd
        the dimension to concatenate with topic vector to infer z_d
        """
        # conv_stride=2 on size 28 got RuntimeError:
        # Given input size: (64x1x1).
        # Calculated output size: (64x0x0).
        # Output size is too small
        # if self.zx_dim != 0: pytorch can generate emtpy tensor,
        # so no need to judge here
        net_infer_zx = LSEncoderConvBnReluPool(
            self.zx_dim, self.i_c, self.i_w, self.i_h,
            conv_stride=1)

        net_infer_zy = EncoderConnectLastFeatLayer2Z(
            self.zy_dim, True, i_c, i_h, i_w, args,
            arg_name="nname", arg_path_name="npath")

        net_infer_zd_topic = EncoderImg2TopicDirZd(args=args,
                                                   num_topics=topic_dim,
                                                   zd_dim=self.zd_dim,
                                                   i_c=self.i_c,
                                                   i_w=self.i_w,
                                                   i_h=self.i_h,
                                                   device=device,
                                                   topic_h_dim=topic_h_dim,
                                                   img_h_dim=img_h_dim)

        super().__init__(net_infer_zd_topic, net_infer_zx, net_infer_zy)


# To remove
class XYDTEncoderConvBnReluPool(XYDTEncoderElevator):
    """
    This class only reimplemented constructor of parent class
    """
    @store_args
    def __init__(self, device, topic_dim, zd_dim, zx_dim, zy_dim,
                 i_c, i_h, i_w,
                 topic_h_dim,
                 img_h_dim,
                 conv_stride,
                 args):
        """
        :param zd_dim:
        :param zx_dim:
        :param zy_dim:
        :param i_c: number of image channels
        :param i_h: image height
        :param i_w: image width
        """
        # conv_stride=2 on size 28 got RuntimeError:
        # Given input size: (64x1x1).
        # Calculated output size: (64x0x0).
        # Output size is too small
        net_infer_zd_topic = EncoderImg2TopicDirZd(args=args,
                                                   num_topics=topic_dim,
                                                   zd_dim=self.zd_dim,
                                                   i_c=self.i_c,
                                                   i_w=self.i_w,
                                                   i_h=self.i_h,
                                                   device=device,
                                                   topic_h_dim=topic_h_dim,
                                                   img_h_dim=img_h_dim)
        # if self.zx_dim != 0: pytorch can generate emtpy tensor,
        # so no need to judge here
        net_infer_zx = LSEncoderConvBnReluPool(
            self.zx_dim, self.i_c, self.i_w, self.i_h,
            conv_stride=conv_stride)
        net_infer_zy = LSEncoderConvBnReluPool(
            self.zy_dim, self.i_c, self.i_w, self.i_h,
            conv_stride=conv_stride)
        super().__init__(net_infer_zd_topic, net_infer_zx, net_infer_zy)
