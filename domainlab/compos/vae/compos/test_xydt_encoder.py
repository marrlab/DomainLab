import torch
from domainlab.compos.vae.compos.encoder_xydt_elevator import XYDTEncoderConvBnReluPool
from domainlab.utils.utils_cuda import get_device

def test_XYDEncoderConvBnReluPool():
    """test"""
    device = get_device(False)
    model = XYDTEncoderConvBnReluPool(device, 3, 8, 8, 8, 3, 64, 64)
    img = torch.rand(2, 3, 64, 64)
    q_topic, topic_q, q_zd, zd_q, q_zx, zx_q, q_zy, zy_q = model(img)
