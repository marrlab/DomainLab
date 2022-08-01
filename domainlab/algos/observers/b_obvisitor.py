import os
import abc
import warnings
import torch

import numpy as np

from domainlab.algos.observers.a_observer import AObVisitor
from domainlab.utils.utils_class import store_args
from domainlab.utils.perf_metrics import PerfClassif
from domainlab.compos.exp.exp_utils import ExpModelPersistVisitor
from domainlab.tasks.task_folder_mk import NodeTaskFolderClassNaMismatch
from domainlab.tasks.task_pathlist import NodeTaskPathListDummy


def pred2file(loader_te, model, device, fa='path_prediction.txt', flag_pred_scalar=False):
    model.eval()
    model_local = model.to(device)
    for i, (x_s, y_s, *_, path) in enumerate(loader_te):
        x_s, y_s = x_s.to(device), y_s.to(device)
        _, prob, *_ = model_local.infer_y_vpicn(x_s)
        # print(path)
        list_pred_list = prob.tolist()
        list_label_list = y_s.tolist()
        if flag_pred_scalar:
            list_pred_list = [np.asarray(pred).argmax() for pred in list_pred_list]
            list_label_list = [np.asarray(label).argmax() for label in list_label_list]
        list_pair_path_pred = list(zip(path, list_label_list, list_pred_list))  # label belongs to data
        with open(fa, 'a') as f:
            for pair in list_pair_path_pred:
                print(str(pair)[1:-1], file=f)  # 1:-1 removes brackets of tuple
    print("prediction saved in file ", fa)


class ObVisitor(AObVisitor):
    """
    Observer + Visitor pattern for model selection
    """
    @store_args
    def __init__(self, exp, model_sel, device):
        """
        observer trainer
        """
        self.host_trainer = None
        self.task = self.exp.task
        self.loader_te = self.exp.task.loader_te
        self.loader_tr = self.exp.task.loader_tr
        self.loader_val = self.exp.task.loader_val
        # Note loader_tr behaves/inherit different properties than loader_te
        self.epo_te = self.exp.args.epo_te
        self.epo = None
        self.metric_te = None
        self.keep_model = self.exp.args.keep_model
        self.perf_metric = PerfClassif(self.task.dim_y)

    def update(self, epoch):
        print("epoch:", epoch)
        self.epo = epoch
        if epoch % self.epo_te == 0:
            metric_tr_pool = self.perf_metric.cal_metrics(self.host_trainer.model, self.loader_tr, self.device)
            print("pooled train domains performance: \n", metric_tr_pool)
            # test set has no domain label, so can be more custom
            metric_te = self.perf_metric.cal_metrics(self.host_trainer.model, self.loader_te, self.device)
            self.metric_te = metric_te
            print("out of domain test performance \n", metric_te)
        if self.model_sel.update():
            print("model selected")
            self.exp.visitor.save(self.host_trainer.model)
            print("persisted")
        return self.model_sel.if_stop()

    def accept(self, trainer):
        """
        accept invitation as a visitor
        """
        self.host_trainer = trainer
        self.model_sel.accept(trainer, self)

    def after_all(self):
        """
        After training is done
        """
        self.exp.visitor.save(self.host_trainer.model, "final")
        model_ld = self.exp.visitor.load()
        model_ld = model_ld.to(self.device)
        model_ld.eval()
        metric_te = self.perf_metric.cal_metrics(model_ld, self.loader_te, self.device)
        print("persisted model performance metric: \n", metric_te)
        self.exp.visitor(metric_te)
        flag_task_folder = isinstance(self.exp.task, NodeTaskFolderClassNaMismatch)
        flag_task_path_list = isinstance(self.exp.task, NodeTaskPathListDummy)
        if flag_task_folder or flag_task_path_list:
            fname4model = self.exp.visitor.model_path
            file_prefix = os.path.splitext(fname4model)[0]  # remove ".csv"
            file_name = file_prefix + "_instance_wise_predictions.txt"
            pred2file(
                self.loader_te, self.host_trainer.model, self.device,
                fa=file_name)

    def clean_up(self):
        """
        to be called by a decorator
        """
        if not self.keep_model:
            try:
                self.exp.visitor.remove("oracle")   # oracle means use out-of-domain test accuracy to select the model
                self.exp.visitor.remove("epoch")    # the last epoch
                # epoch exist to still have a model to evaluate if the training stops in between
                self.exp.visitor.remove("final")
                self.exp.visitor.remove()
            except Exception as e:
                warnings.warn("failed to remove model")
