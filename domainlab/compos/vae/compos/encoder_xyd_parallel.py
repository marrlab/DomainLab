import torch.nn as nn

from domainlab.compos.vae.compos.encoder import LSEncoderConvBnReluPool
from domainlab.compos.vae.compos.encoder_zy import \
    EncoderConnectLastFeatLayer2Z
from domainlab.utils.utils_class import store_args


class XYDEncoderParallel(nn.Module):
    """
    calculate zx, zy, zd vars independently (without order, parallel):
    x->zx, x->zy, x->zd
    """
    def __init__(self, net_infer_zd, net_infer_zx, net_infer_zy):
        super().__init__()
        self.add_module("net_infer_zd", net_infer_zd)
        self.add_module("net_infer_zx", net_infer_zx)
        self.add_module("net_infer_zy", net_infer_zy)

    def forward(self, img):
        """
        encode img into 3 latent variables separately/parallel
        :param img:
        :return q_zd, zd_q, q_zx, zx_q, q_zy, zy_q
        """
        q_zd, zd_q = self.net_infer_zd(img)
        q_zx, zx_q = self.net_infer_zx(img)
        q_zy, zy_q = self.net_infer_zy(img)
        return q_zd, zd_q, q_zx, zx_q, q_zy, zy_q

    def infer_zy_loc(self, tensor):
        """
        Used by VAE model to predict class label
        """
        q_zy, _ = self.net_infer_zy(tensor)
        zy_loc = q_zy.mean
        return zy_loc


class XYDEncoderParallelUser(XYDEncoderParallel):
    """
    This class only reimplemented constructor of parent class
    """
    @store_args
    def __init__(self, net_class_d, net_x, net_class_y):
        super().__init__(net_class_d, net_x, net_class_y)


class XYDEncoderParallelConvBnReluPool(XYDEncoderParallel):
    """
    This class only reimplemented constructor of parent class
    """
    @store_args
    def __init__(self, zd_dim, zx_dim, zy_dim, i_c, i_h, i_w, conv_stride=1):
        """
        :param zd_dim:
        :param zx_dim:
        :param zy_dim:
        :param i_c: number of image channels
        :param i_h: image height
        :param i_w: image width
        """
        # conv_stride=2 on size 28 got RuntimeError:
        # Given input size: (64x1x1).
        # Calculated output size: (64x0x0).
        # Output size is too small
        net_infer_zd = LSEncoderConvBnReluPool(
            self.zd_dim, self.i_c, self.i_w, self.i_h,
            conv_stride=conv_stride)
        # if self.zx_dim != 0:
        # pytorch can generate emtpy tensor, so no need to judge here
        net_infer_zx = LSEncoderConvBnReluPool(
            self.zx_dim, self.i_c, self.i_w, self.i_h,
            conv_stride=conv_stride)
        net_infer_zy = LSEncoderConvBnReluPool(
            self.zy_dim, self.i_c, self.i_w, self.i_h,
            conv_stride=conv_stride)
        super().__init__(net_infer_zd, net_infer_zx, net_infer_zy)


class XYDEncoderParallelAlex(XYDEncoderParallel):
    """
    This class only reimplemented constructor of parent class,
    at the end of the constructor of this class, the parent
    class contructor is called
    """
    @store_args
    def __init__(self, zd_dim, zx_dim, zy_dim, i_c, i_h, i_w, args,
                 conv_stride=1):
        """
        :param zd_dim:
        :param zx_dim:
        :param zy_dim:
        :param i_c: number of image channels
        :param i_h: image height
        :param i_w: image width
        """
        # conv_stride=2 on size 28 got RuntimeError:
        # Given input size: (64x1x1).
        # Calculated output size: (64x0x0).
        # Output size is too small
        net_infer_zd = LSEncoderConvBnReluPool(
            self.zd_dim, self.i_c, self.i_w, self.i_h,
            conv_stride=conv_stride)
        # if self.zx_dim != 0: pytorch can generate emtpy tensor,
        # so no need to judge here
        net_infer_zx = LSEncoderConvBnReluPool(
            self.zx_dim, self.i_c, self.i_w, self.i_h,
            conv_stride=conv_stride)
        net_infer_zy = EncoderConnectLastFeatLayer2Z(self.zy_dim, True,
                                                     i_c, i_h, i_w, args,
                                                     arg_name="nname",
                                                     arg_path_name="npath")
        super().__init__(net_infer_zd, net_infer_zx, net_infer_zy)


class XYDEncoderParallelExtern(XYDEncoderParallel):
    """
    This class only reimplemented constructor of parent class,
    at the end of the constructor of this class, the parent
    class contructor is called
    """
    @store_args
    def __init__(self, zd_dim, zx_dim, zy_dim, args,
                 i_c, i_h, i_w, conv_stride=1):
        """
        :param zd_dim:
        :param zx_dim:
        :param zy_dim:
        """
        net_infer_zd = EncoderConnectLastFeatLayer2Z(self.zd_dim, True,
                                                     i_c, i_h, i_w, args,
                                                     arg_name="nname_dom",
                                                     arg_path_name="npath_dom")
        # if self.zx_dim != 0: pytorch can generate emtpy tensor,
        # so no need to judge zx_dim=0 here
        net_infer_zx = LSEncoderConvBnReluPool(
            self.zx_dim, self.i_c, self.i_w, self.i_h,
            conv_stride=conv_stride)

        net_infer_zy = EncoderConnectLastFeatLayer2Z(self.zy_dim, True,
                                                     i_c, i_h, i_w, args,
                                                     arg_name="nname",
                                                     arg_path_name="npath")
        super().__init__(net_infer_zd, net_infer_zx, net_infer_zy)
