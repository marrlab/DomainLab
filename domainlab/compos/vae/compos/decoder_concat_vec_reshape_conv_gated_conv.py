"""
Bridge Pattern: Separation of interface and implementation.
This class is using one implementation to feed into parent class constructor.
"""
import numpy as np
import torch.nn as nn

from domainlab.compos.nn_zoo.net_gated import Conv2d, GatedConv2d, GatedDense
from domainlab.compos.vae.compos.decoder_concat_vec_reshape_conv import \
    DecoderConcatLatentFcReshapeConv
from domainlab.compos.vae.compos.decoder_losses import NLLPixelLogistic256


class DecoderConcatLatentFCReshapeConvGatedConv(DecoderConcatLatentFcReshapeConv):
    """
    Bridge Pattern: Separation of interface and implementation.
    This class is using implementation to feed into parent class constructor.
    Latent vector re-arranged to image-size directly, then convolute
    to get the textures of the original image
    """
    def __init__(self, z_dim, i_c, i_h, i_w):
        """
        :param z_dim:
        :param list_im_chw: [im_c, im_h, im_w]
        im_c, im_h, im_w: [channel, height, width]
        """
        list_im_chw = [i_c, i_h, i_w]
        cls_fun_nll_p_x = NLLPixelLogistic256
        net_fc_z2flat_img = nn.Sequential(
            GatedDense(z_dim, np.prod(list_im_chw))
        )

        net_conv = nn.Sequential(
            # GatedConv2d
            # input_channels, output_channels, kernel_size,
            # stride, padding, dilation=1, activation=None)
            GatedConv2d(list_im_chw[0], 64, 3, 1, 1),
            GatedConv2d(64, 64, 3, 1, 1),
            # GatedConv2d(64, 64, 3, 1, 1),
            # comment out 2 layers to reduce decoder power
            # GatedConv2d(64, 64, 3, 1, 1),
        )
        #
        # hidden image to mean and variance of each pixel
        # stride(1) and kernel size 1, pad 0
        net_p_x_mean = Conv2d(64, list_im_chw[0], 1, 1, 0,
                              activation=nn.Sigmoid())
        net_p_x_log_var = Conv2d(64, list_im_chw[0], 1, 1, 0,
                                 activation=nn.Hardtanh(min_val=-4.5, max_val=0.))
        super().__init__(z_dim, i_c, i_h, i_w,
                         cls_fun_nll_p_x,
                         net_fc_z2flat_img,
                         net_conv,
                         net_p_x_mean,
                         net_p_x_log_var)
