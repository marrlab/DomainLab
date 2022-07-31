import torch
import torch.nn as nn
import torch.distributions as dist

from domainlab.compos.net_conv import NetConvDense
from domainlab.compos.nn import DenseNet

from domainlab.compos.vae.compos.encoder import LSEncoderDense
from domainlab.compos.vae.compos.encoder_dirichlet import EncoderH2Dirichlet
from domainlab.compos.vae.compos.encoder_domain_topic_img_topic2zd import \
    HEncoderTopicImg2Zd


class EncoderImg2TopicDirZd(nn.Module):
    """
    """
    def __init__(self, i_c, i_h, i_w, num_topics,
                 device,
                 zd_dim,
                 topic_h_dim,
                 img_h_dim,
                 conv_stride,
                 args):
        """__init__.

        :param i_c:
        :param i_h:
        :param i_w:
        :param num_topics:
        :param device:
        :param zd_dim:
        :param topic_h_dim:
        :param img_h_dim: (img->h_img, topic->h_topic)-> q_zd
        the dimension to concatenate with topic vector to infer z_d
        :param conv_stride:
        """
        super().__init__()
        self.device = device
        self.zd_dim = zd_dim
        self.img_h_dim = img_h_dim
        self.topic_h_dim = topic_h_dim

        # image->h_image->[alpha,topic]

        self.add_module("h_layer_img",
                        NetConvDense(i_c, i_h, i_w,
                                     conv_stride=conv_stride,
                                     args=args,
                                     dim_out_h=self.img_h_dim))

        # h_image->[alpha,topic]
        self.add_module("h2dir", EncoderH2Dirichlet(dim_h=self.img_h_dim,
                                                    dim_topic=num_topics,
                                                    device=self.device))

        # [topic, image] -> [h(topic), h(image)] -> [zd_mean, zd_scale]
        self.add_module(
            "imgtopic2zd", HEncoderTopicImg2Zd(
                self.zd_dim, i_c, i_h, i_w,
                num_topics,
                topic_h_dim=self.topic_h_dim,
                img_h_dim=self.img_h_dim,
                conv_stride=conv_stride))


    def forward(self, x):
        """forward.

        :param x:
        """
        # image->h_image
        h_img_dir = self.h_layer_img(x)
        # h_image->alpha
        q_topic, topic_q = self.h2dir(h_img_dir)
        # [topic, image] -> [h(topic), h(image)] -> [zd_mean, zd_scale]
        q_zd, zd_q = self.imgtopic2zd(x, topic_q)
        return q_topic, topic_q, q_zd, zd_q
