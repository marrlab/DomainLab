"""
Upon pixel wise mean and variance
"""
import torch


class NLLPixelLogistic256(object):
    """
    Compute pixel wise negative likelihood of image, given pixel wise mean and variance.
    Pixel intensity is divided into bins of 256 levels.
    p.d.f. is calculated through  c.d.f.(x_{i,j}+bin_size/scale) - c.d.f.(x_{i,j})
    # https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L29
    """
    def __init__(self, reduce_dims=(1, 2, 3), bin_size=1. / 256.):
        """
        :param reduce_dims:
        """
        self.reduce_dims = reduce_dims
        self.bin_size = bin_size

    def __call__(self, tensor, mean, logvar):
        """
        :math: `p(tensor) = prod_{i,j}p(tensor_{i,j}, mu_{i,j}, sigma_{i,j})`
        The reconstruction likelihood should always be summation of all pixels
        :param tensor:
        :param mean:
        :param logvar:
        """
        scale = torch.exp(logvar)
        tensor = (torch.floor(tensor / self.bin_size) * self.bin_size - mean) / scale
        cdf_plus = torch.sigmoid(tensor + self.bin_size/scale)
        cdf_minus = torch.sigmoid(tensor)

        # negative log-likelihood for each pixel
        log_logist_256 = - torch.log(cdf_plus - cdf_minus + 1.e-7)
        nll = torch.sum(log_logist_256, dim=self.reduce_dims)
        # NOTE: pixel NLL should always be summed across the whole image of all channels
        # NOTE: result should be order 1 tensor of dim batch_size
        return nll


def test_NLLPixelLogistic256():
    """
    """
    from libdg.utils.test_img import mk_img
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
