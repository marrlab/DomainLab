"""
use random start to generate adversarial images
"""

from collections import OrderedDict
import torch
from torch import nn

try:
    from backpack import backpack, extend
    from backpack.extensions import Variance
except:
    backpack = None

from domainlab.algos.trainers.train_basic import TrainerBasic


class TrainerFishr(TrainerBasic):
    """
    The goal is to minimize the variance of the domain-level variance of the gradients.
    This aligns the domain-level loss landscapes locally around the final weights, reducing
    inconsistencies across domains.

    For more details, see: Alexandre Ramé, Corentin Dancette, and Matthieu Cord.
        "Fishr: Invariant gradient variances for out-of-distribution generalization."
        International Conference on Machine Learning. PMLR, 2022.
    """
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
            loss = sum(list_loss_erm) + self.aconf.gamma_reg * loss_fishr
            loss.backward()
            self.optimizer.step()
            self.epo_loss_tr += loss.detach().item()
            self.after_batch(epoch, ind_batch)

        flag_stop = self.observer.update(epoch)  # notify observer
        return flag_stop

    def var_grads_and_loss(self, tuple_data_domains_batch):
        """
        Calculate the domain-level variance of the gradients and the layer-wise erm loss.
        Input: a tupel containing lists with the data per domain
        Return: two lists. The first one contains dictionaries with the gradient variances. The keys
        are the layers and the values are tensors. The gradient variances are stored in the tensors.
        The second list contains the losses. Each list entry represents the summed up erm loss of a
        single layer.
        """

        list_dict_var_grads = []
        list_loss_erm = []
        for list_x_y_d_single_domain in tuple_data_domains_batch:  # traverse each domain
            # first dimension of tensor_x is batchsize
            tensor_x, vec_y, vec_d, *_ = tuple(list_x_y_d_single_domain)
            tensor_x, vec_y, vec_d = \
                tensor_x.to(self.device), vec_y.to(self.device), vec_d.to(self.device)
            dict_var_grads_single_domain = self.cal_dict_variance_grads(tensor_x, vec_y)
            list_dict_var_grads.append(dict_var_grads_single_domain)
            loss_erm, *_ = self.model.cal_loss(tensor_x, vec_y, vec_d)
            list_loss_erm.append(loss_erm.sum())  # FIXME: let sum() to be configurable
        # now len(list_dict_var_grads) = (# domains)
        return list_dict_var_grads, list_loss_erm


    def variance_between_dict(self, list_dict_var_paragrad):
        """
        Computes the variance of the domain-level gradient variances, layer-wise.
        Let $v=1/n\\sum_i^n v_i represent the mean across n domains, with
        $$v_i = var(\\nabla_{\\theta}\\ell(x^{(d_i)}, y^{(d_i)}))$$, where $$d_i$$ means data
        coming from domain i. We are interested in $1/n\\sum_(v_i-v)^2=1/n \\sum_i v_i^2 - v^2$.

        Input: list of dictionaries, each dictionary has the structure
        {"layer1": tensor[64, 3, 11, 11],
        "layer2": tensor[8, 3, 5, 5]}.....
        The scalar values in the dictionary are the variances of the gradient of the loss
        w.r.t. the scalar component of the weight tensor for the layer in question, where
        the variance is computed w.r.t. the minibatch of a particular domain.

        Return: dictionary, containing the layers as keys and tensors as values. The variances are
        stored in the tensors as scalars.
        """

        dict_d1 = list_dict_var_paragrad[0]
        # first we determine \\bar(v^2)
        list_dict_var_paragrad_squared = [{key:torch.pow(dict_ele[key], 2) for key in dict_d1}
                                          for dict_ele in list_dict_var_paragrad]
        dict_mean_square_var_paragrad = self.cal_mean_across_dict(list_dict_var_paragrad_squared)

        # now we determine $\\bar(v)^2$
        dict_mean_var_paragrad = \
            {key: torch.mean(torch.stack([ele[key] for ele in list_dict_var_paragrad]), dim=0)
             for key in dict_d1.keys()}
        dict_square_mean_var_paragrad = self.cal_power_single_dict(dict_mean_var_paragrad)

        # now we do \bar(v^2)- (\bar(v))²
        dict_layerwise_var_var_grads = \
            {key:dict_mean_square_var_paragrad[key]-dict_square_mean_var_paragrad[key]
             for key in dict_square_mean_var_paragrad.keys()}
        return dict_layerwise_var_var_grads

    def cal_power_single_dict(self, mdict):
        """
        Calculates the element-wise power of the values in a dictionary, when the values ar tensors.
        Input: dictionary, where the values are tensors.
        Return: dictionary, where the values are tensors. The scalar values of the tensors are the
        element-wise power of the scalars in the input dictionary.
        """

        dict_rst = {key:torch.pow(mdict[key], 2) for key in mdict}
        return dict_rst

    def cal_mean_across_dict(self, list_dict):
        """
        Calculates the mean across several dictionaries.
        Input: list of dictionaries, where the values of each dictionary are tensors.
        Return: dictionary, where the values are tensors. The scalar values of the tensors contain
        the mean across the first dimension of the dictionaries from the list of inputs.
        """

        dict_d1 = list_dict[0]
        dict_mean_var_paragrad = \
            {key: torch.mean(torch.stack([ele[key] for ele in list_dict]), dim=0)
             for key in dict_d1.keys()}
        return dict_mean_var_paragrad

    def cal_dict_variance_grads(self, tensor_x, vec_y):
        """
        Calculates the domain-level variances of the gradients w.r.t. the scalar component of the
        weight tensor for the layer in question, i.e.
        $$v_i = var(\\nabla_{\\theta}\\ell(x^{(d_i)}, y^{(d_i)}))$$, where $$d_i$$ means data
        coming from domain i. The computation is done using the package backpack.

        Input: tensor_x, a tensor, where the first dimension is the batch size and vec_y, which
        is a vector representing the output labels.

        Return: dictionary, where the key is the name for the layer of a neural network and the
        value is the diagonal variance of each scalar component of the gradient of the loss w.r.t.
        the parameter.

        Return Example:
        {"layer1": Tensor[batchsize=32, 64, 3, 11, 11 ]} as a convolution kernel
        """

        loss = self.model.cal_task_loss(tensor_x.clone(), vec_y)
        loss = loss.sum()

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
        return dict_variance
