"""
https://arxiv.org/pdf/2203.10789#page=3.77
"""
import copy
import torch
from torch import nn


class TrainerMiroModelWraper():
    """Mutual-Information Regularization with Oracle"""
    def __init__(self):
        self._features = []
        self._features_ref = []
        self.guest_model = None
        self.ref_model = None
        self.flag_module_found = False

    def get_shapes(self, input_shape):
        # get shape of intermediate features
        self.clear_features()
        with torch.no_grad():
            dummy = torch.rand(*input_shape).to(next(self.guest_model.parameters()).device)
            self.guest_model(dummy)
        shapes = [feat.shape for feat in self._features]
        return shapes

    def accept(self, guest_model, name_feat_layers2extract=None):
        self.guest_model = guest_model
        self.ref_model = copy.deepcopy(guest_model)
        self.register_feature_storage_hook(name_feat_layers2extract)

    def register_feature_storage_hook(self, feat_layers=None):
        # memorize features for each layer in self._feautres list
        if feat_layers is None:
            module = list(self.guest_model.children())[-1]
            module.register_forward_hook(self.hook)
            module_ref = list(self.ref_model.children())[-1]
            module_ref.register_forward_hook(self.hook_ref)
        else:
            for name, module in self.guest_model.named_modules():
                if name in feat_layers:
                    module.register_forward_hook(self.hook)
                    self.flag_module_found = True

            if not self.flag_module_found:
                raise RuntimeError(f"{feat_layers} not found in model!")

            for name, module in self.ref_model.named_modules():
                if name in feat_layers:
                    module.register_forward_hook(self.hook_ref)

    def hook(self, module, input, output):
        self._features.append(output.detach())

    def hook_ref(self, module, input, output):
        self._features_ref.append(output.detach())

    def extract_intermediate_features(self, tensor_x, tensor_y, tensor_d, others=None):
        """
        extract features for each layer of the neural network
        """
        self.clear_features()
        self.guest_model(tensor_x)
        return self._features

    def clear_features(self):
        self._features.clear()

    def cal_feat_layers_ref_model(self, tensor_x, tensor_y, tensor_d, others=None):
        self._features_ref.clear()
        self.ref_model(tensor_x)
        return self._features_ref
