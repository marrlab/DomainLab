"""
operations that all claasification model should have
"""

import abc
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from domainlab.models.a_model import AModel
from domainlab.utils.logger import Logger
from domainlab.utils.perf import PerfClassif
from domainlab.utils.perf_metrics import PerfMetricClassif
from domainlab.utils.utils_class import store_args
from domainlab.utils.utils_classif import get_label_na, logit2preds_vpic
from domainlab.utils.utils_seg import dice_loss

try:
    from backpack import extend
except:
    backpack = None

loss_cross_entropy_extended = extend(nn.CrossEntropyLoss(reduction="none"))


class AModelSeg(AModel, metaclass=abc.ABCMeta):
    """
    operations that all classification model should have
    """

    match_feat_fun_na = "cal_logit_y"

    def extend(self, model):
        super().extend(model)
        self._net_classifier = model.net_classifier

    @property
    def metric4msel(self):
        return "acc"

    @net_classifier.setter
    def net_classifier(self, net_classifier):
        self._net_classifier = net_classifier

    def create_perf_obj(self, task):
        """
        for classification, dimension of target can be quieried from task
        """
        self.perf_metric = PerfMetricClassif(task.dim_y)
        return self.perf_metric

    def cal_perf_metric(self, loader, device):
        """
        classification performance metric
        """
        metric = None
        with torch.no_grad():
            if loader is not None:
                metric = self.perf_metric.cal_metrics(self, loader, device)
                confmat = metric.pop("confmat")
                logger = Logger.get_logger()
                logger.info("scalar performance:")
                logger.info(str(metric))
                logger.debug("confusion matrix:")
                logger.debug(pd.DataFrame(confmat))
                metric["confmat"] = confmat
        return metric

    def evaluate(self, loader_te, device):
        """
        for classification task, use the current model to cal acc
        """
        acc = PerfClassif.cal_acc(self, loader_te, device)
        logger = Logger.get_logger()
        logger.info(f"before training, model accuracy: {acc}")

    def extract_semantic_feat(self, tensor_x):
        """
        flatten the shape of feature tensor from super()
        """
        feat_tensor = super().extract_semantic_feat(tensor_x)
        feat = feat_tensor.reshape(feat_tensor.shape[0], -1)
        return feat

    @store_args
    def __init__(self, **kwargs):
        """
        :param list_str_y: list of fixed order, each element is a class label
        """
        super().__init__()
        for key, value in kwargs.items():
            if key == "net_seg":
                net_seg = value
            
        self.seg_cross_entropy_loss = nn.CrossEntropyLoss()
        self.net_seg = net_seg

    def cal_task_loss(self, tensor_x, tensor_y):
        """
        Calculate the task loss. Used within the `cal_loss` methods of models
        that are subclasses of `AModelClassif`. Cross entropy loss for
        classification is used here by default but could be modified by
        subclasses
        as necessary.

        :param tensor_x: input
        :param tensor_y: label
        :return: task loss
        """
        masks_pred = self.net_seg(tensor_x)
        loss = self.seg_cross_entropy_loss(masks_pred.squeeze(1), tensor_y.float())
        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), tensor_y.float(), multiclass=False)
        return loss

    def _cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        """
        for ERM to adapt to the interface of other regularized learners
        """
        device = tensor_x.device
        bsize = tensor_x.shape[0]
        return [torch.zeros(bsize).to(device)], [0.0]
