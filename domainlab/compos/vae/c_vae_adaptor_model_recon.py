"""
This adaptor couples intensively with the  heavy-weight model class
The model class can be refactored, we do want to use the trained old-version model,
which we only need to change this adaptor class.
"""


class AdaptorReconVAEXYD():
    """
    This adaptor couples intensively with the  heavy-weight model class
    The model class can be refactored, we do want to use the trained old-version model,
    which we only need to change this adaptor class.
    """
    def __init__(self, model):
        self.model = model

    def cal_latent(self, x):
        """
        This method won't be redundant as it will be used several times,
        and most importantly, it **couples** with the models encoder attribute,
        so we only need to change this one time.
        Suppose if model class changes, to use old trained models(we can not retrain them),
        we only need to change this method.
        :param x:
        """
        q_zd, _, q_zx, _, q_zy, _ =  \
            self.model.encoder(x)
        return q_zd, q_zx, q_zy

    def recon_ydx(self, zy, zd, zx, x):
        """
        1. The order of concatnation plays a vital role!
        2. This method won't be redundant as it will be used several times,
        and most importantly, it **couples** with the models encoder attribute,
        so we only need to change this one time.
        Suppose if model class changes, to use old trained models(we can not retrain them),
        we only need to change this method.
        """
        z_concat = self.model.decoder.concat_ydx(zy, zd, zx)
        _, x_mean, _ = self.model.decoder(z_concat, x)
        return x_mean

    def cal_prior_zy(self, vec_y):
        """
        """
        p_zy = self.model.net_p_zy(vec_y)
        return p_zy

    def cal_prior_zd(self, vec_d):
        """
        """
        p_zd = self.model.net_p_zd(vec_d)
        return p_zd
