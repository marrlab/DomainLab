"""
Bridge Pattern: Separation of interface and implementation.
This class is using one implementation to feed into parent class constructor.
"""
import torch
import torch.nn as nn
import numpy as np

from domainlab.compos.vae.compos.decoder_concat_vec_reshape_conv_gated_conv import DecoderConcatLatentFCReshapeConvGatedConv

def test_DecoderConcatLatentFCReshapeConvGatedConv():
    """test"""
    batch_size = 5
    latent_dim = 8
    model = DecoderConcatLatentFCReshapeConvGatedConv(latent_dim, 3, 64, 64)
    vec_z = torch.rand(batch_size, latent_dim)
    x = torch.rand(batch_size, 3, 64, 64)
    re = model.cal_nll(vec_z, x)
    re.sum().backward()
