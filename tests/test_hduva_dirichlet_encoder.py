"""
end to end test
"""
import torch

from domainlab.compos.vae.compos.encoder_dirichlet import EncoderH2Dirichlet


def test_unit_encoder_dirichlet():
    encoder_dirichlet = EncoderH2Dirichlet(dim_topic=3, device=torch.device("cpu"))
    feat_hidden_uniform01 = torch.rand(32, 3)  # batchsize 32
    encoder_dirichlet(feat_hidden_uniform01)
    feat_hidden_normal = torch.normal(0, 1, size=(32, 3))
    encoder_dirichlet(feat_hidden_normal)
    feat_hidden_uniform_big = feat_hidden_uniform01 * 1e9
    encoder_dirichlet(feat_hidden_uniform_big)
    feat_hidden_uniform_small = feat_hidden_uniform01 * 1e-9
    encoder_dirichlet(feat_hidden_uniform_small)
