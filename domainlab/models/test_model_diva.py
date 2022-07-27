from domainlab.models.model_diva import ModelDIVA
from domainlab.compos.vae.utils_request_chain_builder import RequestVAEBuilderCHW, VAEChainNodeGetter
from domainlab.utils.test_img import mk_rand_xyd
from domainlab.utils.utils_classif import mk_dummy_label_list_str


def test_fun():
    im_h = 64
    y_dim = 10
    dim_d_tr = 3
    batch_size = 5

    request = RequestVAEBuilderCHW(3, im_h, im_h)
    node = VAEChainNodeGetter(request)()

    list_str_y = mk_dummy_label_list_str("class", y_dim)
    list_d_tr = mk_dummy_label_list_str("domain", dim_d_tr)

    model = ModelDIVA(node, zd_dim=8, zy_dim=8, zx_dim=8, gamma_d=1.0,
                      gamma_y=1.0,
                      list_str_y=list_str_y, list_d_tr=list_d_tr)
    imgs, y_s, d_s = mk_rand_xyd(im_h, y_dim, dim_d_tr, batch_size)
    one_hot, mat_prob, label, confidence, na = model.infer_y_vpicn(imgs)
    model(imgs, y_s, d_s)
