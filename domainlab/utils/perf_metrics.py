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
        with torch.no_grad():
            for i, (x_s, y_s, *_) in enumerate(loader_te):
                x_s, y_s = x_s.to(device), y_s.to(device)
                prob, *_ = model_local.infer_y_vpicn(x_s)
                _, pred_label = torch.max(prob, 1)
                self.acc.update(pred_label, y_s)
                self.precision.update(pred_label, y_s)
                self.recall.update(pred_label, y_s)
                self.specificity.update(pred_label, y_s)
                self.f1_score.update(pred_label, y_s)
                self.auroc.update(prob, y_s)
                self.confmat.update(pred_label, y_s)
                if i > max_batches:
                    break

        acc_y = self.acc.compute()
        precision_y = self.precision.compute()
        recall_y = self.recall.compute()
        specificity_y = self.specificity.compute()
        f1_score_y = self.f1_score.compute()
        auroc_y = self.auroc.compute()
        confmat_y = self.confmat.compute()
        return {"acc": acc_y, "precision": precision_y, "recall": recall_y,
                "psecificity": specificity_y, "f1": f1_score_y,
                "auroc": auroc_y,
                "confmat": confmat_y}
