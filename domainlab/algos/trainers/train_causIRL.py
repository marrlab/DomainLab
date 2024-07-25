"""
Alex, Xudong
"""
import numpy as np
import torch
from domainlab.algos.trainers.train_basic import TrainerBasic


class TrainerCausIRL(TrainerBasic):
    """
    causal matching
    """
    def my_cdist(self, x1, x2):
        """
        distance for Gaussian
        """
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        """
        kernel for MMD
        """
        dist = self.my_cdist(x, y)
        K = torch.zeros_like(dist)

        for g in gamma:
            K.add_(torch.exp(dist.mul(-g)))

        return K

    def mmd(self, x, y):
        """
        maximum mean discrepancy
        """
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff


    def tr_batch(self, tensor_x, tensor_y, tensor_d, others, ind_batch, epoch):
        """
        optimize neural network one step upon a mini-batch of data
        """
        self.kernel_type = "gaussian"
        self.before_batch(epoch, ind_batch)
        tensor_x, tensor_y, tensor_d = (
            tensor_x.to(self.device),
            tensor_y.to(self.device),
            tensor_d.to(self.device),
        )
        self.optimizer.zero_grad()

        features = self.get_model().extract_semantic_feat(tensor_x)

        pos_batch_break = np.random.randint(0, tensor_x.shape[0])
        first = features[:pos_batch_break]
        second = features[pos_batch_break:]
        if len(first) > 1 and len(second) > 1:
            penalty = torch.nan_to_num(self.mmd(first, second))
        else:
            penalty = torch.tensor(0)
        loss = self.cal_loss(tensor_x, tensor_y, tensor_d, others)
        loss = loss + penalty
        loss.backward()
        self.optimizer.step()
        self.after_batch(epoch, ind_batch)
        self.counter_batch += 1
