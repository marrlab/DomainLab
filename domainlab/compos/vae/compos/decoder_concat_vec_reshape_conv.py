"""
decoder which takes concatenated latent representation
"""
import torch
import torch.nn as nn


class DecoderConcatLatentFcReshapeConv(nn.Module):
    """
    Latent vector re-arranged to image-size directly,
    then convolute to get the textures of the original image
    """
    def __init__(self, z_dim, i_c, i_h, i_w,
                 cls_fun_nll_p_x,
                 net_fc_z2flat_img,
                 net_conv,
                 net_p_x_mean,
                 net_p_x_log_var):
        """
        :param z_dim:
        :param list_im_chw: [channel, height, width]
        """
        self.z_dim = z_dim
        self.cls_fun_nll_p_x = cls_fun_nll_p_x
        self.list_im_chw = [i_c, i_h, i_w]
        super().__init__()
        self.add_module("net_fc_z2flat_img", net_fc_z2flat_img)
        self.add_module("net_conv", net_conv)
        self.add_module("net_p_x_mean", net_p_x_mean)
        self.add_module("net_p_x_log_var", net_p_x_log_var)

    def cal_p_x_pars_loc_scale(self, vec_z):
        """compute mean and variance of each pixel
        :param z:
        """
        h_flat = self.net_fc_z2flat_img(vec_z)
        # reshape to image
        h_img = h_flat.view(-1, self.list_im_chw[0], self.list_im_chw[1], self.list_im_chw[2])
        h_img_conv = self.net_conv(h_img)
        # pixel must be positive: enforced by sigmoid activation
        x_mean = self.net_p_x_mean(h_img_conv)  # .view(-1, np.prod(self.list_im_chw))
        # remove the saturated part of sigmoid
        x_mean = torch.clamp(x_mean, min=0.+1./512., max=1.-1./512.)
        # negative values
        x_logvar = self.net_p_x_log_var(h_img_conv)  # .view(-1, np.prod(self.list_im_chw))
        return x_mean, x_logvar

    def concat_ydx(self, zy, zd, zx):
        z_concat = torch.cat((zy, zd, zx), 1)
        return z_concat

    def concat_ytdx(self, zy, topic, zd, zx):
        z_concat = torch.cat((zy, topic, zd, zx), 1)
        return z_concat

    def forward(self, z, img):
        """
        :param z:
        :param img:
        """
        x_mean, x_logvar = self.cal_p_x_pars_loc_scale(z)
        nll = self.cls_fun_nll_p_x()(img, x_mean, x_logvar)
        return nll, x_mean, x_logvar
