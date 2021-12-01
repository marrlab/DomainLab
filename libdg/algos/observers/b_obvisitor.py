import os
import abc
import torch

from libdg.algos.observers.a_observer import AObVisitor
from libdg.utils.utils_class import store_args
from libdg.utils.perf import PerfClassif
from libdg.compos.exp.exp_utils import ExpModelPersistVisitor


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
        self.loader_te = self.exp.task.loader_te
        self.loader_tr = self.exp.task.loader_tr
        self.epo_te = self.exp.args.epo_te
        self.epo = None
        self.acc_te = None
        self.keep_model = self.exp.args.keep_model

    def update(self, epoch):
        print("epoch:", epoch)
        self.epo = epoch
        if epoch % self.epo_te == 0:
            acc_tr_pool = PerfClassif.cal_acc(self.host_trainer.model, self.loader_tr, self.device)
            print("pooled train domain acc: ", acc_tr_pool)
            acc_te = PerfClassif.cal_acc(self.host_trainer.model, self.loader_te, self.device)
            self.acc_te = acc_te
            print("out of domain test acc: ", acc_te)
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
        acc_te = PerfClassif.cal_acc(model_ld, self.loader_te, self.device)
        print("persisted model acc: ", acc_te)
        self.exp.visitor(acc_te)

    def clean_up(self):
        """
        to be called by a decorator
        """
        if not self.keep_model:
            self.exp.visitor.remove("epoch")
            # epoch exist to still have a model to evaluate if the training stops in between
            self.exp.visitor.remove("final")
            self.exp.visitor.remove()
            self.exp.visitor.remove("oracle")
