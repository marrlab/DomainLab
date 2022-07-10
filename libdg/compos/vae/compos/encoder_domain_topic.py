import torch
import torch.nn as nn
import torch.distributions as dist

from libdg.compos.net_conv import NetConvDense
from libdg.compos.nn import DenseNet

from libdg.compos.vae.compos.encoder import LSEncoderDense
from libdg.compos.vae.compos.encoder_dirichlet import EncoderH2Dirichlet


class HEncoderTopicImg2Zd(nn.Module):
    """
    sandwich encoder: (img, s)->zd
    """
    def __init__(self, zd_dim, i_c, i_h, i_w, num_topics,
                 topic_h_dim, img_h_dim, conv_stride=1):
        """
        topic_h_dim, img_h_dim: (img->h_img, topic->h_topic)-> q_zd
        """
        super().__init__()
        self.zd_dim = zd_dim
        self.topic_h_dim = topic_h_dim
        self.img_h_dim = img_h_dim
        # image->h_img
        self.add_module("h_layer_img",
                        NetConvDense(i_c, i_h, i_w,
                                     conv_stride=conv_stride, dim_out_h=self.img_h_dim))
        # topic->h_topic
        self.add_module("h_layer_topic",
                        DenseNet(input_flat_size=num_topics, out_hidden_size=self.topic_h_dim))
        # [h_img, h_topic] -> zd
        self.add_module("img_topic_h2zd", LSEncoderDense(dim_input=self.img_h_dim+self.topic_h_dim,
                                                         z_dim=self.zd_dim))

    def forward(self, x, topic):
        # image->h_img
        h_img = self.h_layer_img(x)
        # topic->h_topic
        h_topic = self.h_layer_topic(topic)
        #
        h_img_topic = torch.cat((h_img, h_topic), 1)   # FIXME: order of concatnation
        q_zd, zd_q = self.img_topic_h2zd(h_img_topic)
        return q_zd, zd_q



class EncoderImg2TopicDirZd(nn.Module):
    """
    """
    def __init__(self, i_c, i_h, i_w, num_topics,
                 device,
                 zd_dim,
                 topic_h_dim=512,
                 img_h_dim=512,
                 conv_stride=1):
        super().__init__()
        self.device = device
        self.zd_dim = zd_dim
        self.img_h_dim = img_h_dim
        self.topic_h_dim = topic_h_dim

        # image->h_image->[alpha,topic]

        self.add_module("h_layer_img",
                        NetConvDense(i_c, i_h, i_w,
                                     conv_stride=conv_stride, dim_out_h=self.img_h_dim))

        # h_image->[alpha,topic]
        self.add_module("h2dir", EncoderH2Dirichlet(dim_h=self.img_h_dim,
                                                    dim_topic=num_topics,
                                                    device=self.device))

        # [topic, image] -> [h(topic), h(image)] -> [zd_mean, zd_scale]
        self.add_module("imgtopic2zd", HEncoderTopicImg2Zd(self.zd_dim, i_c, i_h, i_w,
                                                           num_topics,
                                                           topic_h_dim=self.topic_h_dim,
                                                           img_h_dim=self.img_h_dim,
                                                           conv_stride=conv_stride))


    def forward(self, x):
        # image->h_image
        h_img_dir = self.h_layer_img(x)
        # h_image->alpha
        q_topic, topic_q = self.h2dir(h_img_dir)
        # [topic, image] -> [h(topic), h(image)] -> [zd_mean, zd_scale]
        q_zd, zd_q = self.imgtopic2zd(x, topic_q)
        return q_topic, topic_q, q_zd, zd_q
