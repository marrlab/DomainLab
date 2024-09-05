"""
Alex, Xudong
"""
import numpy as np
import torch
from domainlab.algos.trainers.train_basic import TrainerBasic


class TrainerCausalIRL(TrainerBasic):
    """
    causal matching
    """
    def my_cdist(self, x1, x2):
        """
        distance for Gaussian
        """
        # along the last dimension
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        # x_2_norm is [batchsize, 1]
        # matrix multiplication (2nd, 3rd) and addition to first argument
        # X1[batchsize, dimfeat] * X2[dimfeat, batchsize)
        # alpha: Scaling factor for the matrix product (default: 1)
        # x2_norm.transpose(-2, -1) is row vector
        # x_1_norm is column vector
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y):
        """
        kernel for MMD
        """
        gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]
        dist = self.my_cdist(x, y)
        tensor = torch.zeros_like(dist)
        for g in gamma:
            tensor.add_(torch.exp(dist.mul(-g)))
        return tensor

    def mmd(self, x, y):
        """
        maximum mean discrepancy
        """
        kxx = self.gaussian_kernel(x, x).mean()
        kyy = self.gaussian_kernel(y, y).mean()
        kxy = self.gaussian_kernel(x, y).mean()
        return kxx + kyy - 2 * kxy

    def tr_epoch(self, epoch):
        list_loaders = list(self.dict_loader_tr.values())
        loaders_zip = zip(*list_loaders)
        self.model.train()
        self.model.convert4backpack()
        self.epo_loss_tr = 0

        for ind_batch, tuple_data_domains_batch in enumerate(loaders_zip):
            self.optimizer.zero_grad()
            list_dict_var_grads, list_loss_erm = self.var_grads_and_loss(tuple_data_domains_batch)
            dict_layerwise_var_var_grads = self.variance_between_dict(list_dict_var_grads)
            dict_layerwise_var_var_grads_sum = \
                {key: val.sum() for key, val in dict_layerwise_var_var_grads.items()}
            loss_fishr = sum(dict_layerwise_var_var_grads_sum.values())
            loss = sum(list_loss_erm) + get_gamma_reg(self.aconf, self.name) * loss_fishr
            loss.backward()
            self.optimizer.step()
            self.epo_loss_tr += loss.detach().item()
            self.after_batch(epoch, ind_batch)

        flag_stop = self.observer.update(epoch)  # notify observer
        return flag_stop


