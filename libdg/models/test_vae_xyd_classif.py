import torch
import torch.nn as nn
import torch.distributions as dist
from torch.nn import functional as F

from libdg.utils.utils_class import store_args
from libdg.utils.utils_classif import logit2preds_vpic, get_label_na
from libdg.models.a_model_classif import AModelClassif
from libdg.models.model_vae_xyd_classif import VAEXYDClassif
from libdg.compos.vae.utils_request_chain_builder import RequestVAEBuilderCHW, VAEChainNodeGetter
from libdg.utils.test_img import mk_rand_xyd
from libdg.utils.utils_classif import mk_dummy_label_list_str


def testvae():
    y_dim = 10
    d_dim = 3
    batch_size = 5
    ims = 28
    imc = 3
    request = RequestVAEBuilderCHW(imc, ims, ims)
    node = VAEChainNodeGetter(request)()

    list_str_y = mk_dummy_label_list_str("class", y_dim)
    list_str_d = mk_dummy_label_list_str("domain", d_dim)

    model = VAEXYDClassif(node, zd_dim=8, zy_dim=8, zx_dim=8, list_str_y=list_str_y, list_str_d=list_str_d)
    imgs, y, d = mk_rand_xyd(ims, y_dim, d_dim, batch_size)
    one_hot, mat_prob, ind, confi, na = model.infer_y(imgs)
    confi.numpy()
