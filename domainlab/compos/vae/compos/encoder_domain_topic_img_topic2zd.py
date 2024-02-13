import torch
import torch.nn as nn

from domainlab.compos.vae.compos.encoder import LSEncoderLinear
from domainlab.compos.zoo_nn import FeatExtractNNBuilderChainNodeGetter


class EncoderSandwichTopicImg2Zd(nn.Module):
    """
    sandwich encoder: (img, s)->zd
    """

    def __init__(self, zd_dim, isize, num_topics, img_h_dim, args):
        """
        num_topics, img_h_dim: (img->h_img, topic->h_topic)-> q_zd
        :param img_h_dim: (img->h_img, topic->h_topic)-> q_zd
        the dimension to concatenate with topic vector to infer z_d
        """
        super().__init__()
        self.zd_dim = zd_dim
        self.img_h_dim = img_h_dim

        net_builder = FeatExtractNNBuilderChainNodeGetter(
            args=args,
            arg_name_of_net="nname_encoder_sandwich_x2h4zd",
            arg_path_of_net="npath_encoder_sandwich_x2h4zd",
        )()

        # image->h_img
        self.add_module(
            "layer_img2h4zd",
            net_builder.init_business(
                dim_out=self.img_h_dim,
                flag_pretrain=True,
                remove_last_layer=False,
                isize=isize,
                args=args,
            ),
        )

        # [h_img, h_topic] -> zd
        self.add_module(
            "encoder_cat_topic_img_h2zd",
            LSEncoderLinear(dim_input=self.img_h_dim + num_topics, z_dim=self.zd_dim),
        )

    def forward(self, img, vec_topic):
        """forward.

        :param x:
        :param topic:
        """
        # image->h_img
        h_img = self.layer_img2h4zd(img)
        h_topic = vec_topic
        h_img_topic = torch.cat((h_img, h_topic), 1)
        q_zd, zd_q = self.encoder_cat_topic_img_h2zd(h_img_topic)
        return q_zd, zd_q
