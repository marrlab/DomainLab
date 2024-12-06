"""
use random start to generate adversarial images
"""
import torch
from torch import autograd
from torch.nn import functional as F
from domainlab.algos.trainers.train_irm import TrainerIRM


class TrainerIRMSepDom(TrainerIRM):
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
                    self.cal_reg_loss(tensor_x, tensor_y, tensor_d, others)
                list_domain_reg += list_1ele_loss_irm
            loss = torch.sum(torch.stack(list_domain_loss_erm)) + \
                self.aconf.gamma_reg * torch.sum(torch.stack(list_domain_reg))
            loss.backward()
            self.optimizer.step()
            self.epo_loss_tr += loss.detach().item()
            self.after_batch(epoch, ind_batch)

        flag_stop = self.observer.update(epoch, flag_info)  # notify observer
        return flag_stop
