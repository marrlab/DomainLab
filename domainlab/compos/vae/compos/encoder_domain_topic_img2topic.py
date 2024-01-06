import torch.nn as nn

from domainlab.compos.vae.compos.encoder_dirichlet import EncoderH2Dirichlet
from domainlab.compos.zoo_nn import FeatExtractNNBuilderChainNodeGetter


class EncoderImg2TopicDistri(nn.Module):
    """
    image to topic distribution  (not image to topic hidden representation
    used by another path)
    """
    def __init__(self, isize, num_topics,
                 device,
                 args):
        """__init__.

        :param isize:
        :param num_topics:
        :param device:
        """
        super().__init__()
        self.device = device

        # image->h_topic->(batchnorm, exp)[alpha,topic]

        net_builder = FeatExtractNNBuilderChainNodeGetter(
            args=args,
            arg_name_of_net="nname_encoder_x2topic_h",
            arg_path_of_net="npath_encoder_x2topic_h")()

        self.add_module("layer_img2hidden",
                        net_builder.init_business(
                            flag_pretrain=True,
                            isize=isize,
                            remove_last_layer=False,
                            dim_out=num_topics,
                            args=args))

        # h_image->[alpha,topic]
        self.add_module("layer_hidden2dirichlet",
                        EncoderH2Dirichlet(
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
