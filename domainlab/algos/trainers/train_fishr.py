"""
use random start to generate adversarial images
"""
import torch
import copy
from torch.autograd import Variable
import torch.nn as nn
from collections import OrderedDict

try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad, Variance
except:
    backpack = None

from domainlab.algos.trainers.train_basic import TrainerBasic

_bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))

class TrainerFishr(TrainerBasic):
    """
    Trainer Domain Invariant Adversarial Learning
    """
    def tr_epoch(self, epoch):
        self.model.train()
        self.model.convert4backpack()
        self.epo_loss_tr = 0
        for ind_batch, (tensor_x, vec_y, vec_d, *_) in enumerate(self.loader_tr):
            tensor_x, vec_y, vec_d = \
                tensor_x.to(self.device), vec_y.to(self.device), vec_d.to(self.device)
            self.optimizer.zero_grad()
            loss_erm = self.model.cal_loss(tensor_x, vec_y, vec_d)  # @FIXME
            loss_fishr = self.cal_fishr(tensor_x, vec_y, vec_d)
            loss = loss_erm.sum() + self.aconf.gamma_reg * loss_fishr
            loss.backward()
            self.optimizer.step()
            self.epo_loss_tr += loss.detach().item()
            self.after_batch(epoch, ind_batch)
        flag_stop = self.observer.update(epoch)  # notify observer
        return flag_stop

    def variance_between_dict(self, list_dict_var_paragrad):
        """
        the input of this function is a list of dictionaries, each dictionary
        has the structure
        {"layer1": tensor[64, 3, 11, 11],
         "layer2": tensor[8, 3, 5, 5]}.....
         the scalar value of this dictionary means the variance of the gradient of the loss
         w.r.t. the scalar component of the weight tensor for the layer in question, where
         the variance is computed w.r.t. the minibatch of a particular domain.

         Choose a specific scalar component of the gradient tensor w.r.t. scalar weight $$\\theta$$
         $$v_i = var(\\nabla_{\\theta}\\ell(x^{(d_i)}, y^{(d_i)}))$$, where $$d_i$$ means data
         coming from domain i, and $var$ means the variance.
         Let $v=1/n\\sum_i v_i represent the mean across n domains
         We are interested in $1/n\\sum_(v_i-v)^2=1/n \\sum_i v_i^2 - v^2$
        """
        import torch
        dict_d1 = {f"layer{i}": torch.rand(3,3) for i in range(5)}
        dict_d2 = {f"layer{i}": torch.rand(3,3) for i in range(5)}
        dict_d3 = {f"layer{i}": torch.rand(3,3) for i in range(5)}
        list_dict = [dict_d1, dict_d2, dict_d3]
        dict_key_list = {key: [ele[key] for ele in list_dict] for key in dict_d1.keys()}
        tensor_stack = torch.stack(dict_key_list["layer1"])
        torch.mean(tensor_stack, dim=0)

        # use 1
        dict_mean = {key: torch.mean(torch.stack([ele[key] for ele in list_dict]), dim=0) for key in dict_d1.keys()}
        # e
        {torch.pow(dict_d1[key], 2) for key in dict_d1}

        list_dict_pow = [{torch.pow(dict_ele[key], 2) for key in dict_d1} for dict_ele in list_dict]

        dict_mean_v2 = self.cal_mean_across_dict(list_dict_pow)
        dict_v_pow = self.cal_power_single_dict(dict_mean)

        dict_fishr = {dict_mean_v2[key]-dict_v_pow[key] for key in dict_v_pow.keys()}

    def cal_power_single_dict(self, mdict):
        """
        """
        dict_rst = {torch.pow(mdict[key], 2) for key in mdict}
        return dict_rst

    def cal_mean_across_dict(list_dict):
        """
        """
        dict_d1 = list_dict[0]
        dict_mean = {key: torch.mean(torch.stack([ele[key] for ele in list_dict]), dim=0) for key in dict_d1.keys()}
        return dict_mean

    def cal_fishr(self, tensor_x, vec_y, vec_d):
        """
        use backpack
        """
        loss = self.model.cal_task_loss(tensor_x.clone(), vec_y)

        with backpack(Variance()):
            loss.backward(
                inputs=list(self.model.parameters()), retain_graph=True, create_graph=True
            )

        for name, param in self.model.named_parameters():
            print(name)
            print(".grad.shape:             ", param.variance.shape)

        dict_variance = OrderedDict(
            [(name, weights.variance.clone())
             for name, weights in self.model.named_parameters()
             ])
        return 0
