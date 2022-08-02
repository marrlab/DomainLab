import torch
from domainlab.compos.vae.compos.encoder_domain_topic import EncoderSandwichTopicImg2Zd, EncoderImg2TopicDirZd


def test_TopicImg2Zd():
    model = EncoderSandwichTopicImg2Zd(zd_dim=64, i_c=3, i_h=64, i_w=64,
                                num_topics=5, topic_h_dim=1024, img_h_dim=1024,
                                conv_stride=1)
    x = torch.rand(20, 3, 64, 64)
    topic = torch.rand(20, 5)
    q_zd, zd_q = model(x, topic)


def test_EncoderImg2TopicDir_Zd():
    from domainlab.utils.utils_cuda import get_device
    device = get_device(flag_no_cu=False)
    model = EncoderImg2TopicDirZd(3, 64, 64, num_topics=5,
                                  img_h_dim=512,
                                  topic_h_dim=512,
                                  device=device,
                                  zd_dim=64)
    x = torch.rand(20, 3, 64, 64)
    q_topic, topic_q, q_zd, zd_q = model(x)

    # bigger image
    model = EncoderImg2TopicDirZd(3, 224, 224, num_topics=5,
                                  img_h_dim=512,
                                  topic_h_dim=512,
                                  device=device,
                                  zd_dim=64)
    x = torch.rand(20, 3, 224, 224)
    q_topic, topic, q_zd, zd_q = model(x)
