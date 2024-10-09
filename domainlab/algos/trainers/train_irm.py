"""
use random start to generate adversarial images
"""
import torch
from torch import autograd
from torch.nn import functional as F
from domainlab.algos.trainers.train_basic import TrainerBasic


class TrainerIRM(TrainerBasic):
    """
    IRMv1 split a minibatch into half, and use an unbiased estimate of the
    squared gradient norm via inner product
    $$\\delta_{w|w=1} \\ell(w\\dot \\Phi(X^{e, i}), Y^{e, i})$$
    of dimension dim(Grad)
    with
    $$\\delta_{w|w=1} \\ell(w\\dot \\Phi(X^{e, j}), Y^{e, j})$$
    of dimension dim(Grad)
    For more details, see section 3.2 and Appendix D of :
    Arjovsky et al., “Invariant Risk Minimization.”
    """
    def tr_epoch(self, epoch, flag_info=False):
        list_loaders = list(self.dict_loader_tr.values())
        loaders_zip = zip(*list_loaders)
        self.model.train()
        self.epo_loss_tr = 0

        for ind_batch, tuple_data_domains_batch in enumerate(loaders_zip):
            self.optimizer.zero_grad()
            list_domain_loss_erm = []
            list_domain_reg = []
            for batch_domain_e in tuple_data_domains_batch:
                tensor_x, tensor_y, tensor_d, *others = batch_domain_e
                tensor_x, tensor_y, tensor_d = \
                    tensor_x.to(self.device), tensor_y.to(self.device), \
                    tensor_d.to(self.device)
                list_domain_loss_erm.append(
                    self.model.cal_task_loss(tensor_x, tensor_y))
                list_1ele_loss_irm, _ = \
                    self._cal_reg_loss(tensor_x, tensor_y, tensor_d, others)
                list_domain_reg += list_1ele_loss_irm
            loss = torch.sum(torch.stack(list_domain_loss_erm)) + \
                self.aconf.gamma_reg * torch.sum(torch.stack(list_domain_reg))
            loss.backward()
            self.optimizer.step()
            self.epo_loss_tr += loss.detach().item()
            self.after_batch(epoch, ind_batch)

        flag_stop = self.observer.update(epoch, flag_info)  # notify observer
        return flag_stop

    def _cal_phi(self, tensor_x):
        logits = self.model.cal_logit_y(tensor_x)
        return logits

    def _cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        """
        Let trainer behave like a model, so that other trainer could use it
        """
        _ = tensor_d
        _ = others
        y = tensor_y
        with torch.enable_grad():
            phi = self._cal_phi(tensor_x)
            dummy_w_scale = torch.tensor(1.).to(tensor_x.device).requires_grad_()
            loss_1 = F.cross_entropy(phi[::2] * dummy_w_scale, y[::2])
            loss_2 = F.cross_entropy(phi[1::2] * dummy_w_scale, y[1::2])
            grad_1 = autograd.grad(loss_1, [dummy_w_scale], create_graph=True)[0]
            grad_2 = autograd.grad(loss_2, [dummy_w_scale], create_graph=True)[0]
            loss_irm_scalar = torch.sum(grad_1 * grad_2)  # scalar
            loss_irm_tensor = loss_irm_scalar.expand(tensor_x.shape[0])
            return [loss_irm_tensor], [self.aconf.gamma_reg]
