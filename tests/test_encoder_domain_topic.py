import torch
from domainlab.compos.vae.compos.encoder_domain_topic import EncoderSandwichTopicImg2Zd
from domainlab.arg_parser import mk_parser_main


def test_TopicImg2Zd():
    parser = mk_parser_main()
    args = parser.parse_args([
                                "--te_d", "9", "--dpath",
                                "zout", "--split", "0.8"])
    args.nname_encoder_sandwich_x2h4zd = "conv_bn_pool_2"
    model = EncoderSandwichTopicImg2Zd(
        zd_dim=64, isize=(3,64,64),
        num_topics=5, img_h_dim=1024,
        args=args)
    x = torch.rand(20, 3, 64, 64)
    topic = torch.rand(20, 5)
    _, _ = model(x, topic)
