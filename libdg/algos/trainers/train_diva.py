from libdg.algos.trainers.a_trainer import TrainerClassif
import torch.optim as optim


class TrainerDIVA(TrainerClassif):
    def __init__(self, model, task, observer, device, aconf):
        super().__init__(model, task, observer, device, aconf)
        self.optimizer = optim.Adam(self.model.parameters(), lr=aconf.lr)
        self.epo_loss_tr = None
        self.epo_lc_y = None
        self.epo_loss_recon = None

    def tr_epoch(self, epoch):
        self.model.train()
        self.model.warm_up_beta(epoch)
        self.epo_loss_tr = 0
        self.epo_lc_y = 0
        self.epo_loss_recon = 0
        for _, (tensor_x, vec_y, vec_d) in enumerate(self.loader_tr):
            tensor_x, vec_y, vec_d = \
                tensor_x.to(self.device), vec_y.to(self.device), vec_d.to(self.device)
            self.optimizer.zero_grad()
            loss, loss_recon_x,  \
            zd_p_minus_zd_q, zx_p_minus_zx_q, zy_p_minus_zy_q, \
            lc_y, lc_d = \
                self.model(tensor_x, vec_y, vec_d)
            loss = loss.sum()
            loss.backward()
            self.optimizer.step()
            self.epo_loss_tr += loss.detach().item()
            self.epo_loss_recon += loss_recon_x.sum().detach().item()
            self.epo_lc_y += lc_y.sum().detach().item()
        flag_stop = self.observer.update(epoch)  # notify observer
        return flag_stop


def test_trainer_diva():
    import torch
    from observers.a_observer import ObserverDummy
    from libdg.models.model_diva import ModelDIVA
    from libdg.utils.utils_classif import mk_dummy_label_list_str
    from libdg.compos.vae.utils_request_chain_builder import RequestVAEBuilderCHW, VAEChainNodeGetter
    from libdg.dsets.dset_poly_domains_mnist_color_default import DsetMNISTColorMix
    dset = DsetMNISTColorMix(n_domains=3, path="zout")
    y_dim = 10
    d_dim = 3
    list_str_y = mk_dummy_label_list_str("class", y_dim)
    list_str_d = mk_dummy_label_list_str("domain", d_dim)
    request = RequestVAEBuilderCHW(3, 28, 28)
    node = VAEChainNodeGetter(request)()
    model = ModelDIVA(node, zd_dim=8, zy_dim=8, zx_dim=8, d_dim=d_dim, list_str_y=list_str_y, list_str_d=list_str_d, gamma_d=1.0, gamma_y=1.0)
    loader = torch.utils.data.DataLoader(dset, batch_size=60, shuffle=True)
    trainer = TrainerDIVA(model, ObserverDummy(), loader_tr=loader, loader_te=loader, lr=1e-3)
    trainer.before_tr()
    trainer.tr_epoch(0)
    trainer.before_tr()
