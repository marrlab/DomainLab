import torch
from observers.a_observer import ObserverDummy
from domainlab.models.model_diva import ModelDIVA
from domainlab.utils.utils_classif import mk_dummy_label_list_str
from domainlab.compos.vae.utils_request_chain_builder import RequestVAEBuilderCHW, VAEChainNodeGetter
from domainlab.dsets.dset_poly_domains_mnist_color_default import DsetMNISTColorMix
from domainlab.algos.trainers.train_visitor import TrainerVisitor


def test_trainer_diva():
    dset = DsetMNISTColorMix(n_domains=3, path="zout")
    y_dim = 10
    d_dim = 3
    list_str_y = mk_dummy_label_list_str("class", y_dim)
    list_str_d = mk_dummy_label_list_str("domain", d_dim)
    request = RequestVAEBuilderCHW(3, 28, 28)
    node = VAEChainNodeGetter(request)()
    model = ModelDIVA(node, zd_dim=8, zy_dim=8, zx_dim=8, d_dim=d_dim, list_str_y=list_str_y, list_str_d=list_str_d, gamma_d=1.0, gamma_y=1.0)
    loader = torch.utils.data.DataLoader(dset, batch_size=60, shuffle=True)
    trainer = TrainerVisitor(model, ObserverDummy(), loader_tr=loader, loader_te=loader, lr=1e-3)
    trainer.before_tr()
    trainer.tr_epoch(0)
    trainer.before_tr()
