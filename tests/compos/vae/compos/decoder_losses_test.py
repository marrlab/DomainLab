"""
Upon pixel wise mean and variance
"""
import torch
from domainlab.utils.test_img import mk_img
from domainlab.compos.vae.compos.decoder_losses import NLLPixelLogistic256

def test_NLLPixelLogistic256():
    """
    """
    img = mk_img(3)
    mean = mk_img(3)
    var = mk_img(3)/100.  # SNR
    log_var = torch.log(var)
    nll = NLLPixelLogistic256()(img, mean, log_var)
    assert len(nll.shape) == 1
    assert nll.shape[0] == img.shape[0]
    NLLPixelLogistic256()(img, img, log_var)  # use identical image as mean, NLL only depends on var
    NLLPixelLogistic256()(img, img, log_var/0.1)  # big variance lead to small NLL loss
    NLLPixelLogistic256()(img, img, log_var/100.) # small variance lead to big NLL loss
    NLLPixelLogistic256()(img, img, log_var/1000.)
