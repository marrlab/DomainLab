"""
operations that all kinds of models should have
"""

import abc

from torch import nn
from domainlab import g_list_model_penalized_reg_agg


class AModel(nn.Module, metaclass=abc.ABCMeta):
    """
    operations that all models (classification, segmentation, seq2seq)
    """
    def __init__(self):
        super().__init__()
        self._decoratee = None
        self.list_d_tr = None
        self.visitor = None
        self._net_invar_feat = None

    def extend(self, model):
        """
        extend the loss of the decoratee
        """
        self._decoratee = model
        self.reset_feature_extractor(model.net_invar_feat)

    @property
    def metric4msel(self):
        """
        metric for model selection
        """
        raise NotImplementedError

    @property
    def multiplier4task_loss(self):
        """
        the multiplier for task loss is default to 1 except for vae family models
        """
        return 1.0

    def cal_loss(self, tensor_x, tensor_y, tensor_d=None, others=None):
        """
        calculate the loss
        """
        list_loss, list_multiplier = self.cal_reg_loss(tensor_x, tensor_y, tensor_d, others)
        loss_reg = self.list_inner_product(list_loss, list_multiplier)
        loss_task_alone = self.cal_task_loss(tensor_x, tensor_y)
        loss_task = self.multiplier4task_loss * loss_task_alone
        return loss_task + loss_reg, list_loss, loss_task_alone

    def list_inner_product(self, list_loss, list_multiplier):
        """
        compute inner product between list of regularization loss and multiplier
        - the length of the list is the number of regularizers
        - for each element of the list: the first dimension of the tensor is mini-batch
        return value of list_inner_product should keep the minibatch structure, thus aggregation
        here only aggregate along the list
        """
        list_tuple = zip(list_loss, list_multiplier)
        list_penalized_reg = [mtuple[0]*mtuple[1] for mtuple in list_tuple]
        tensor_batch_penalized_loss = g_list_model_penalized_reg_agg(list_penalized_reg)
        # return value of list_inner_product should keep the minibatch structure, thus aggregation
        # here only aggregate along the list
        return tensor_batch_penalized_loss

    @abc.abstractmethod
    def cal_task_loss(self, tensor_x, tensor_y):
        """
        Calculate the task loss

        :param tensor_x: input
        :param tensor_y: label
        :return: task loss
        """

    @abc.abstractmethod
    def _cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        """
        task independent regularization loss for domain generalization
        """

    def cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        """
        task independent regularization loss for domain generalization
        """
        loss_reg, mu = self._extend_loss(
            tensor_x, tensor_y, tensor_d, others)
        loss_reg_, mu_ = self._cal_reg_loss(
            tensor_x, tensor_y, tensor_d, others)
        if loss_reg is not None:
            return loss_reg_ + loss_reg, mu_ + mu
        return loss_reg_, mu_

    def _extend_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        """
        combine losses from two models
        """
        if self._decoratee is not None:
            return self._decoratee.cal_reg_loss(
                tensor_x, tensor_y, tensor_d, others)
        return None, None

    def forward(self, tensor_x, tensor_y, tensor_d, others=None):
        """forward.

        :param x:
        :param y:
        :param d:
        """
        return self.cal_loss(tensor_x, tensor_y, tensor_d, others)

    def extract_semantic_feat(self, tensor_x):
        """
        extract semantic feature (not domain feature), note that
        extract semantic feature is an action, it is more general than
        calling a static network(module)'s forward function since 
        there are extra action like reshape the tensor 
        """
        if self._decoratee is not None:
            return self._decoratee.extract_semantic_feat(tensor_x)
        feat = self._net_invar_feat(tensor_x)
        return feat

    @property
    def net_invar_feat(self):
        """
        if exist, return a neural network for extracting invariant features
        """
        return self._net_invar_feat

    def reset_feature_extractor(self, net):
        """
        for two models to share the same neural network, the feature extractor has to be reset
        for classification, both feature extractor and classifier has to be reset
        """
        # note if net is None, which means the decoratee does not have net_invar_feat (can be
        # because there is tensor reshape during forward pass, which can not be represented
        # by a static neural network, in this case, we simply set self._net_invar_feat to be
        # None
        self._net_invar_feat = net
        self.reset_aux_net()

    def reset_aux_net(self):
        """
        after feature extractor being reset, the input dim of other networks like domain
        classification will also change (for commandline usage only)
        """
        # by default doing nothing

    def save(self, suffix=None):
        """
        persist model to disk
        """
        if self.visitor is None:
            return
        self.visitor.save(self, suffix)
        return

    def load(self, suffix=None):
        """
        load model from disk
        """
        if self.visitor is None:
            return None
        return self.visitor.load(suffix)

    def set_saver(self, visitor):
        self.visitor = visitor

    def dset_decoration_args_algo(self, args, ddset):
        """
        decorate dataset to get extra entries in load item, for instance, jigen need permutation index
        this parent class function delegate decoration to its decoratee
        """
        if self._decoratee is not None:
            return self._decoratee.dset_decoration_args_algo(args, ddset)
        return ddset
