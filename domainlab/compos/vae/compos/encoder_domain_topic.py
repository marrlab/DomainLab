import torch.nn as nn

from domainlab.compos.vae.compos.encoder_domain_topic_img2topic import \
    EncoderImg2TopicDistri
from domainlab.compos.vae.compos.encoder_domain_topic_img_topic2zd import \
    EncoderSandwichTopicImg2Zd


class EncoderImg2TopicDirZd(nn.Module):
    """
    """
    def __init__(self, i_c, i_h, i_w, num_topics,
                 device,
                 zd_dim,
                 topic_h_dim,
                 img_h_dim,
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
        """
        super().__init__()
        self.device = device
        self.zd_dim = zd_dim
        self.img_h_dim = img_h_dim
        self.topic_h_dim = topic_h_dim

        self.add_module("net_img2topicdistri",
                        EncoderImg2TopicDistri(
                            i_c, i_h, i_w, num_topics,
                            self.img_h_dim,
                            device,
                            args))

        # [topic, image] -> [h(topic), h(image)] -> [zd_mean, zd_scale]
        self.add_module(
            "imgtopic2zd", EncoderSandwichTopicImg2Zd(
                self.zd_dim, i_c, i_h, i_w,
                num_topics,
                topic_h_dim=self.topic_h_dim,
                img_h_dim=self.img_h_dim,
                args=args))

    def forward(self, img):
        """forward.
        :param img:
        """
        # image->h_image
        # h_image->alpha
        q_topic, topic_q = self.net_img2topicdistri(img)
        # [topic, image] -> [h(topic), h(image)] -> [zd_mean, zd_scale]
        q_zd, zd_q = self.imgtopic2zd(img, topic_q)
        return q_topic, topic_q, q_zd, zd_q
