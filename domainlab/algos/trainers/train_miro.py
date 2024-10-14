"""
author: Kakao Brain.
# https://arxiv.org/pdf/2203.10789#page=3.77
# [aut] xudong, alexej
"""

import torch
from torch import nn
from domainlab.algos.trainers.train_basic import TrainerBasic
from domainlab.algos.trainers.train_miro_utils import \
    MeanEncoder, VarianceEncoder
from domainlab.algos.trainers.train_miro_model_wraper import \
    TrainerMiroModelWraper


class TrainerMiro(TrainerBasic):
    """Mutual-Information Regularization with Oracle"""
    def before_tr(self):
        self.model_wraper = TrainerMiroModelWraper()
        self.model_wraper.accept(self.model)
        self.mean_encoders = None
        self.var_encoders = None
        super().before_tr()
        shapes = self.model_wraper.get_shapes(self.input_tensor_shape)
        self.mean_encoders = nn.ModuleList([
            MeanEncoder(shape) for shape in shapes
        ])
        self.var_encoders = nn.ModuleList([
            VarianceEncoder(shape) for shape in shapes
        ])

    def _cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        # list_batch_inter_feat_new are features for each layer
        list_batch_inter_feat_new = \
            self.model_wraper.extract_intermediate_features(
                tensor_x, tensor_y, tensor_d, others)

        # reference model
        with torch.no_grad():
            list_batch_inter_feat_ref = self.model_wraper.cal_feat_layers_ref_model(
                tensor_x, tensor_y, tensor_d, others)
            # dim(list_batch_inter_feat_ref)=[size_batch, dim_feat]
        if self.mean_encoders is None:
            device = tensor_x.device
            return [torch.zeros(tensor_x.shape[0]).to(device)], [self.aconf.gamma_reg]

        reg_loss = 0.
        num_layers = len(self.mean_encoders)
        device = tensor_x.device
        for ind_layer in range(num_layers):
            # layerwise mutual information regularization
            mean_encoder = self.mean_encoders[ind_layer].to(device)
            feat = list_batch_inter_feat_new[ind_layer]
            feat = feat.to(device)
            mean = mean_encoder(feat)
            var_encoder = self.var_encoders[ind_layer].to(device)
            var = var_encoder(feat)
            mean_ref = list_batch_inter_feat_ref[ind_layer]
            mean_ref = mean_ref.to(device)
            vlb = (mean - mean_ref).pow(2).div(var) + var.log()
            reg_loss += vlb.mean(axis=-1) / 2.
        return [reg_loss], [self.aconf.gamma_reg]
