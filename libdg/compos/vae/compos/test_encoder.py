import torch
from libdg.compos.vae.compos.encoder_xyd_parallel import XYDEncoderParallelConvBnReluPool


def test_XYDEncoderConvBnReluPool():
    """test"""
    model = XYDEncoderParallelConvBnReluPool(8, 8, 8, 3, 64, 64)
    img = torch.rand(2, 3, 64, 64)
    q_zd, zd_q, q_zx, zx_q, q_zy, zy_q = model(img)
