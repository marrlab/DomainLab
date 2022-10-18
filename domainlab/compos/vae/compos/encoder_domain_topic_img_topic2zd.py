import torch
import torch.nn as nn

from domainlab.compos.vae.compos.encoder import LSEncoderDense
from domainlab.compos.zoo_nn import FeatExtractNNBuilderChainNodeGetter


class EncoderSandwichTopicImg2Zd(nn.Module):
    """
    sandwich encoder: (img, s)->zd
    """
    def __init__(self, zd_dim, i_c, i_h, i_w, num_topics,
                 topic_h_dim, img_h_dim, args):
        """
        topic_h_dim, img_h_dim: (img->h_img, topic->h_topic)-> q_zd
        :param img_h_dim: (img->h_img, topic->h_topic)-> q_zd
        the dimension to concatenate with topic vector to infer z_d
        """
        super().__init__()
        self.zd_dim = zd_dim
        self.topic_h_dim = num_topics
        self.img_h_dim = img_h_dim

        net_builder = FeatExtractNNBuilderChainNodeGetter(
            args=args,
            arg_name_of_net="nname_encoder_sandwich_layer_img2h4zd",
            arg_path_of_net="npath_encoder_sandwich_layer_img2h4zd")()  # @FIXME

        # image->h_img
        self.add_module("layer_img2h4zd", net_builder.init_business(
                            dim_out=self.img_h_dim,
                            flag_pretrain=True,
                            remove_last_layer=False,
                            i_c=i_c, i_h=i_h, i_w=i_w, args=args))

        # topic->h_topic
        # @FIXME: do we need topic to h_topic instead of simplying using topic?
        # REMOVED: self.add_module("h_layer_topic",
        # REMOVED:  DenseNet(
        # REMOVED:  input_flat_size=num_topics,
        # REMOVED:  out_hidden_size=self.topic_h_dim))

        # [h_img, h_topic] -> zd
        self.add_module("encoder_cat_topic_img_h2zd",
                        LSEncoderDense(
                            dim_input=self.img_h_dim+self.topic_h_dim,
                            z_dim=self.zd_dim))

    def forward(self, img, vec_topic):
        """forward.

        :param x:
        :param topic:
        """
        # image->h_img
        h_img = self.layer_img2h4zd(img)
        # topic->h_topic
        # REMOVE: h_topic = self.h_layer_topic(topic)
        h_topic = vec_topic
        # @FIXME: order of concatnation
        h_img_topic = torch.cat((h_img, h_topic), 1)
        q_zd, zd_q = self.encoder_cat_topic_img_h2zd(h_img_topic)
        return q_zd, zd_q
