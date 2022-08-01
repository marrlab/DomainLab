"""Classification Performance"""
import numpy as np
import torch
from torchmetrics.classification import Accuracy, Precision, Recall, Specificity, F1Score, AUROC, AUC, ConfusionMatrix


class PerfClassif():
    """Classification Performance metrics"""
    def __init__(self, num_classes, average='macro', device='cpu'):
        super().__init__()
        
        self.acc = Accuracy(num_classes, average).to(device)
        self.precision = Precision(num_classes, average).to(device)
        self.recall = Recall(num_classes, average).to(device)
        self.f1_score = F1Score(num_classes, average).to(device)
        self.auroc = AUROC(num_classes, average).to(device)
        self.specificity = Specificity(num_classes, average).to(device)
        self.confmat = ConfusionMatrix(num_classes).to(device)

    def cal_metrics(self, model, loader_te, device, max_batches=None):
        """
        :param model:
        :param loader_te:
        :param device: for final test, GPU can be used
        :param max_batches:
                maximum number of iteration for data loader, used to
                probe performance with less computation burden.
                default None, which means to traverse the whole dataset
        """
        model.eval()
        model_local = model.to(device)
        if max_batches is None:
            max_batches = len(loader_te)
        list_vec_preds, list_vec_labels = [], []
        with torch.no_grad():
            for i, (x_s, y_s, *_) in enumerate(loader_te):
                x_s, y_s = x_s.to(device), y_s.to(device)
                pred, *_ = model_local.infer_y_vpicn(x_s)
                _, pred_label = torch.max(pred, 1)
                
                acc_ = self.acc.update(pred_label, y_s)
                precision_ = self.precision.update(pred_label, y_s)
                recall_ = self.recall.update(pred_label, y_s)
                specificity_ = self.specificity.update(pred_label, y_s)
                f1_score_ = self.f1_score.update(pred_label, y_s)
                auroc_ = self.auroc.update(pred, y_s)
                confmat_ = self.confmat.update(pred_label, y_s)
                if i > max_batches:
                    break
        
        acc_y = self.acc.compute()
        precision_y = self.precision.compute()
        recall_y = self.recall.compute()
        specificity_y = self.specificity.compute()
        f1_score_y = self.f1_score.compute()
        auroc_y = self.auroc.compute()
        confmat_y = self.confmat.compute()
        
        return acc_y, precision_y, recall_y, specificity_y, f1_score_y, auroc_y, confmat_y
