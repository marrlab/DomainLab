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

    def tr_batch(self, tensor_x, tensor_y, tensor_d, others, ind_batch, epoch):
        """
        optimize neural network one step upon a mini-batch of data
        """
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
