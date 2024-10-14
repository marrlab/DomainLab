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
        shapes = self.model_wraper.get_shapes()
        self.mean_encoders = nn.ModuleList([
            MeanEncoder(shape) for shape in shapes
        ])
        self.var_encoders = nn.ModuleList([
            VarianceEncoder(shape) for shape in shapes
        ])

    def _cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        # batch_inter_feat_new are features for each layer
        batch_inter_feat_new = \
            self.model_wraper.extract_intermediate_features(tensor_x)

        # reference model
        with torch.no_grad():
            batch_inter_feat_ref = self.model_wraper.cal_feat_layers_ref_model(tensor_x)
            # dim(batch_inter_feat_ref)=[size_batch, dim_feat]

        reg_loss = 0.
        num_layers = len(self.mean_encoders)
        for ind_layer in num_layers:
            # layerwise mutual information regularization
            mean = self.mean_encoders[ind_layer](batch_inter_feat_new)
            var = self.var_encoders[ind_layer](batch_inter_feat_new)
            vlb = (mean - batch_inter_feat_ref).pow(2).div(var) + var.log()
            reg_loss += vlb.mean() / 2.
        return reg_loss
