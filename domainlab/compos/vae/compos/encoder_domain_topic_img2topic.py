import torch.nn as nn

from domainlab.compos.vae.compos.encoder_dirichlet import EncoderH2Dirichlet
from domainlab.compos.zoo_nn import FeatExtractNNBuilderChainNodeGetter


class EncoderImg2TopicDistri(nn.Module):
    """
    image to topic distribution  (not image to topic hidden representation
    used by another path)
    """
    def __init__(self, i_c, i_h, i_w, num_topics,
                 img_h_dim,
                 device,
                 args):
        """__init__.

        :param i_c:
        :param i_h:
        :param i_w:
        :param num_topics:
        :param device:
        """
        super().__init__()
        self.device = device
        self.img_h_dim = img_h_dim

        # image->h_image->[alpha,topic]

        # @FIXME:
        net_builder = FeatExtractNNBuilderChainNodeGetter(
            args=args,
            arg_name_of_net="nname_topic_distrib_img2topic",
            arg_path_of_net="npath_topic_distrib_img2topic")()  # @FIXME

        self.add_module("layer_img2hidden",
                        net_builder.init_business(
                            flag_pretrain=True,
                            remove_last_layer=False,
                            dim_out=self.img_h_dim,
                            i_c=i_c, i_h=i_h, i_w=i_w, args=args))

        # self.add_module("layer_img2hidden",
        #                NetConvDense(i_c, i_h, i_w,
        #                             conv_stride=conv_stride,
        #                             args=args,
        #                             dim_out_h=self.img_h_dim))

        # h_image->[alpha,topic]
        self.add_module("layer_hidden2dirichlet",
                        EncoderH2Dirichlet(
                            dim_h=self.img_h_dim,
                            dim_topic=num_topics,
                            device=self.device))

    def forward(self, x):
        """forward.

        :param x:
        """
        # image->h_image
        h_img_dir = self.layer_img2hidden(x)
        # h_image->alpha
        q_topic, topic_q = self.layer_hidden2dirichlet(h_img_dir)
        return q_topic, topic_q
