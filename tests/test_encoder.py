import torch

from domainlab.compos.vae.compos.encoder import LSEncoderConvBnReluPool
from domainlab.compos.vae.compos.encoder_xyd_parallel import (
    XYDEncoderParallelConvBnReluPool,
)


def test_XYDEncoderConvBnReluPool():
    """test"""
    model = XYDEncoderParallelConvBnReluPool(8, 8, 8, 3, 64, 64)
    img = torch.rand(2, 3, 64, 64)
    _, _, _, _, _, _ = model(img)


def test_LSEncoderConvStride1BnReluPool():
    """test"""
    from domainlab.utils.test_img import mk_img

    img_size = 28
    img = mk_img(img_size)
    model = LSEncoderConvBnReluPool(
        z_dim=8, i_channel=3, i_h=img_size, i_w=img_size, conv_stride=1
    )
    q_zd, zd_q = model(img)
    q_zd.mean
    q_zd.scale
    zd_q
