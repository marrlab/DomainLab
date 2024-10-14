"""
https://arxiv.org/pdf/2203.10789#page=3.77
"""
import copy
import torch
from torch import nn


def get_shapes(model, input_shape):
    # get shape of intermediate features
    with torch.no_grad():
        dummy = torch.rand(1, *input_shape).to(next(model.parameters()).device)
        _, feats = model(dummy, ret_feats=True)
        shapes = [f.shape for f in feats]
    return shapes


class TrainerMiroModelWraper():
    """Mutual-Information Regularization with Oracle"""
    def __init__(self):
        self._features = []
        self._features_ref = []
        self.guest_model = None
        self.ref_model = None

    def get_shapes(self, input_shape):
        return get_shapes(self.guest_model, input_shape)

    def accept(self, guest_model):
        self.guest_model = guest_model
        self.ref_model = copy.deepcopy(guest_model)
        self.register_feature_storage_hook()

    def register_feature_storage_hook(self, feat_layers):
        # memorize features for each layer in self._feautres list
        for name, module in self.guest_model.named_modules():
            if name in feat_layers:
                module.register_forward_hook(self.hook)

        for name, module in self.ref_model.named_modules():
            if name in feat_layers:
                module.register_forward_hook(self.hook_ref)

    def hook(self, module, input, output):
        self._features.append(output)

    def hook_ref(self, module, input, output):
        self._features_ref.append(output)

    def extract_intermediate_features(self, tensor_x):
        """
        extract features for each layer of the neural network
        """
        self.clear_features()
        self.guest_model(tensor_x)
        return self._features

    def clear_features(self):
        self._features.clear()

    def cal_feat_layers_ref_model(self, tensor_x):
        self._features_ref.clear()
        self.ref_model(tensor_x)
