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
from domainlab.utils.utils_class import store_args
from domainlab.utils.utils_classif import get_label_na, logit2preds_vpic
from domainlab.utils.perf import PerfClassif
from domainlab.utils.perf_metrics import PerfMetricClassif
from domainlab.utils.logger import Logger


class AModelClassif(AModel, metaclass=abc.ABCMeta):
    """
    operations that all classification model should have
    """
    match_feat_fun_na = "cal_logit_y"

    @property
    def metric4msel(self):
        return "acc"

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
        by default, use the logit as extracted feature if the current method
        is not being overriden by child class
        """
        return self.cal_logit_y(tensor_x)

    @abc.abstractmethod
    def cal_logit_y(self, tensor_x):
        """
        calculate the logit for softmax classification
        """

    @store_args
    def __init__(self, list_str_y=None):
        """
        :param list_str_y: list of fixed order, each element is a class label
        """
        super().__init__()
        self.list_str_y = list_str_y
        self.perf_metric = None
        self.loss4gen_adv = nn.KLDivLoss(size_average=False)

    def infer_y_vpicn(self, tensor):
        """
        :param tensor: input
        :return: vpicn
            v: vector of one-hot class label,
            p: vector of probability,
            i: class label index,
            c: confidence: maximum probability,
            n: list of name of class
        """
        with torch.no_grad():
            logit_y = self.cal_logit_y(tensor)
        vec_one_hot, prob, ind, confidence = logit2preds_vpic(logit_y)
        na_class = get_label_na(ind, self.list_str_y)
        return vec_one_hot, prob, ind, confidence, na_class

    @property
    def dim_y(self):
        """
        the class embedding dimension
        """
        return len(self.list_str_y)

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
        logit_y = self.cal_logit_y(tensor_x)
        if (tensor_y.shape[-1] == 1) | (len(tensor_y.shape) == 1):
            y_target = tensor_y
        else:
            _, y_target = tensor_y.max(dim=1)
        lc_y = F.cross_entropy(logit_y, y_target, reduction="none")
        # cross entropy always return a scalar, no need for inside instance reduction
        return lc_y

    def pred2file(self, loader_te, device, filename,
                  metric_te,
                  spliter="#"):
        """
        pred2file dump predicted label to file as sanity check
        """
        self.eval()
        model_local = self.to(device)
        logger = Logger.get_logger()
        for _, (x_s, y_s, *_, path4instance) in enumerate(loader_te):
            x_s, y_s = x_s.to(device), y_s.to(device)
            _, prob, *_ = model_local.infer_y_vpicn(x_s)
            list_pred_prob_list = prob.tolist()
            list_target_list = y_s.tolist()
            list_target_scalar = [np.asarray(label).argmax() for label in list_target_list]
            tuple_zip = zip(path4instance, list_target_scalar, list_pred_prob_list)
            list_pair_path_pred = list(tuple_zip)
            with open(filename, 'a', encoding="utf8") as handle_file:
                for list4one_obs_path_prob_target in list_pair_path_pred:
                    list_str_one_obs_path_target_predprob = [
                        str(ele) for ele in list4one_obs_path_prob_target]
                    str_line = (" "+spliter+" ").join(list_str_one_obs_path_target_predprob)
                    str_line = str_line.replace("[", "")
                    str_line = str_line.replace("]", "")
                    print(str_line, file=handle_file)
        logger.info(f"prediction saved in file {filename}")
        file_acc = self.read_prediction_file(filename, spliter)
        acc_metric_te = metric_te['acc']
        flag1 = math.isclose(file_acc, acc_metric_te, rel_tol=1e-9, abs_tol=0.01)
        acc_raw1 = PerfClassif.cal_acc(self, loader_te, device)
        acc_raw2 = PerfClassif.cal_acc(self, loader_te, device)
        flag_raw_consistency = math.isclose(acc_raw1, acc_raw2, rel_tol=1e-9, abs_tol=0.01)
        flag2 = math.isclose(file_acc, acc_raw1, rel_tol=1e-9, abs_tol=0.01)
        if not (flag1 & flag2 & flag_raw_consistency):
            str_info = f"inconsistent acc: \n" \
                       f"prediction file acc generated using the current model is {file_acc} \n" \
                       f"input torchmetric acc to the current function: {acc_metric_te} \n" \
                       f"raw acc 1 {acc_raw1} \n" \
                       f"raw acc 2 {acc_raw2} \n"
            raise RuntimeError(str_info)
        return file_acc

    def read_prediction_file(self, filename, spliter):
        """
        check if the written fiel could calculate acc
        """
        with open(filename, 'r', encoding="utf8") as handle_file:
            list_lines = [line.strip().split(spliter) for line in handle_file]
        count_correct = 0
        for line in list_lines:
            list_prob = [float(ele) for ele in line[2].split(",")]
            if np.array(list_prob).argmax() == int(line[1]):
                count_correct += 1
        acc = count_correct / len(list_lines)
        logger = Logger.get_logger()
        logger.info(f"accuracy from prediction file {acc}")
        return acc

    def cal_loss_gen_adv(self, x_natural, x_adv, vec_y):
        """
        calculate loss function for generation of adversarial images
        """
        x_adv.requires_grad_()
        with torch.enable_grad():
            logits_adv = self.cal_logit_y(x_adv)
            logits_natural = self.cal_logit_y(x_natural)
            prob_adv = F.log_softmax(logits_adv, dim=1)
            prob_natural = F.softmax(logits_natural, dim=1)
            loss_adv_gen_task = self.cal_task_loss(x_adv, vec_y)
            loss_adv_gen = self.loss4gen_adv(prob_adv, prob_natural)
        return loss_adv_gen + loss_adv_gen_task.sum()

    def _cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        """
        for ERM to adapt to the interface of other regularized learners
        """
        device = tensor_x.device
        bsize = tensor_x.shape[0]
        return [torch.zeros(bsize, 1).to(device)], [0.0]
