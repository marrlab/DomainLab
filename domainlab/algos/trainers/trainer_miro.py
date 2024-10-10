# author: Kakao Brain.
# https://arxiv.org/pdf/2203.10789#page=3.77
# [aut] xudong, alexej

import torch
import torch.nn as nn
import torch.nn.functional as F
from domainlab.algos.trainers.train_basic import TrainerBasic


def get_shapes(model, input_shape):
    # get shape of intermediate features
    with torch.no_grad():
        dummy = torch.rand(1, *input_shape).to(next(model.parameters()).device)
        _, feats = model(dummy, ret_feats=True)
        shapes = [f.shape for f in feats]

    return shapes


class TrainerMiro(TrainerBasic):
    """Mutual-Information Regularization with Oracle"""
    def register(self):
        # memorize features for each layer in self._feautres list
        for n, m in self.network.named_modules():
                    if n in feat_layers:
                        m.register_forward_hook(self.hook)

    def _init(self):
        shapes = get_shapes(self.pre_featurizer, self.input_shape)
        self.mean_encoders = nn.ModuleList([
            MeanEncoder(shape) for shape in shapes
        ])
        self.var_encoders = nn.ModuleList([
            VarianceEncoder(shape) for shape in shapes
        ])

    def extract_intermediate_features(self, tensor_x):
        """
        extract features for each layer of the neural network
        """

    def _cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        # inter_layer_feats are features for each layer
        feat, inter_layer_feats = self.extract_intermediate_features(tensor_x)
        # orcale model
        with torch.no_grad():
            _, pre_feats = self.pre_featurizer(all_x, ret_feats=True)
            # dim(pre_feats)=[size_batch, dim_feat]

        reg_loss = 0.
        for f, pre_f, mean_enc, var_enc in misc.zip_strict(
            inter_layer_feats, pre_feats, self.mean_encoders, self.var_encoders
        ):
            # mutual information regularization
            mean = mean_enc(f)
            var = var_enc(f)
            vlb = (mean - pre_f).pow(2).div(var) + var.log()
            reg_loss += vlb.mean() / 2.

        return reg_loss


def zip_strict(*iterables):
    """strict version of zip. The length of iterables should be same.

    NOTE yield looks non-reachable, but they are required.
    """
    # For trivial cases, use pure zip.
    if len(iterables) < 2:
        return zip(*iterables)

    # Tail for the first iterable
    first_stopped = False
    def first_tail():
        nonlocal first_stopped
        first_stopped = True
        return




class URResNet(torch.nn.Module):
    """ResNet + FrozenBN + IntermediateFeatures
    """

    def __init__(self, input_shape, hparams, preserve_readout=False, freeze=None, feat_layers=None):
        super().__init__()

        self._features = []
        self.feat_layers = self.build_feature_hooks(feat_layers, block_names)


    def hook(self, module, input, output):
        self._features.append(output)

    def build_feature_hooks(self, feats, block_names):
        assert feats in ["stem_block", "block"]

        if feats is None:
            return []

        # build feat layers
        if feats.startswith("stem"):
            last_stem_name = block_names["stem"][-1]
            feat_layers = [last_stem_name]
        else:
            feat_layers = []

        for name, module_names in block_names.items():
            if name == "stem":
                continue

            module_name = module_names[-1]
            feat_layers.append(module_name)

        #  print(f"feat layers = {feat_layers}")

        for n, m in self.network.named_modules():
            if n in feat_layers:
                m.register_forward_hook(self.hook)

        return feat_layers
