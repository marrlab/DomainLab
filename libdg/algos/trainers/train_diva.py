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
                tensor_x.to(self.device), vec_y.to(self.device), \
                vec_d.to(self.device)
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
