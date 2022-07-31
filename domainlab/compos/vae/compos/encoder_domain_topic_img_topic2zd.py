import torch
import torch.nn as nn
import torch.distributions as dist

from domainlab.compos.net_conv import NetConvDense
from domainlab.compos.nn import DenseNet

from domainlab.compos.vae.compos.encoder import LSEncoderDense
from domainlab.compos.vae.compos.encoder_dirichlet import EncoderH2Dirichlet


class HEncoderTopicImg2Zd(nn.Module):
    """
    sandwich encoder: (img, s)->zd
    """
    def __init__(self, zd_dim, i_c, i_h, i_w, num_topics,
                 topic_h_dim, img_h_dim,
                 conv_stride, args):
        """
        topic_h_dim, img_h_dim: (img->h_img, topic->h_topic)-> q_zd
        :param img_h_dim: (img->h_img, topic->h_topic)-> q_zd
        the dimension to concatenate with topic vector to infer z_d
        """
        super().__init__()
        self.zd_dim = zd_dim
        self.topic_h_dim = topic_h_dim
        self.img_h_dim = img_h_dim
        # image->h_img
        self.add_module("h_layer_img",
                        NetConvDense(
                            i_c, i_h, i_w,
                            conv_stride=conv_stride,
                            dim_out_h=self.img_h_dim,
                            args=args))
        # topic->h_topic
        # FIXME: do we need topic to h_topic instead of simplying using topic?
        # self.add_module("h_layer_topic",
        #                DenseNet(
        #                    input_flat_size=num_topics,
        #                    out_hidden_size=self.topic_h_dim))
        # [h_img, h_topic] -> zd
        self.add_module("img_topic_h2zd",
                        LSEncoderDense(
                            dim_input=self.img_h_dim+self.topic_h_dim,
                            z_dim=self.zd_dim))

    def forward(self, x, topic):
        """forward.

        :param x:
        :param topic:
        """
        # image->h_img
        h_img = self.h_layer_img(x)
        # topic->h_topic
        # h_topic = self.h_layer_topic(topic)
        h_topic = h_img
        h_img_topic = torch.cat((h_img, h_topic), 1)   # FIXME: order of concatnation
        q_zd, zd_q = self.img_topic_h2zd(h_img_topic)
        return q_zd, zd_q
