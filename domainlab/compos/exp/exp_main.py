import datetime
import os
import shutil

from torch.utils.data import Subset
import torch.utils.data as data_utils
import numpy as np

from domainlab.algos.zoo_algos import AlgoBuilderChainNodeGetter
from domainlab.compos.exp.exp_utils import AggWriter
from domainlab.tasks.zoo_tasks import TaskChainNodeGetter
from domainlab.dsets.utils_data import plot_ds
from domainlab.utils.sanity_check import SanityCheck

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # debug


class Exp():
    """
    Exp is combination of Task, Algorithm, and Configuration (including random seed)
    """
    def __init__(self, args, task=None, visitor=AggWriter):
        """
        :param args:
        :param task:
        """
        self.task = task
        if task is None:
            self.task = TaskChainNodeGetter(args)()
            if args.san_check:
                sancheck = SanityCheck(args, self.task)
                sancheck.dataset_sanity_check()
        self.task.init_business(args)
        self.args = args
        self.visitor = visitor(self)
        algo_builder = AlgoBuilderChainNodeGetter(self.args)()  # request
        self.trainer = algo_builder.init_business(self)
        self.epochs = self.args.epos
        self.epoch_counter = 1

    def execute(self):
        """
        train model
        check performance by loading persisted model
        """
        t_0 = datetime.datetime.now()
        print('\n Experiment start at :', str(t_0))
        t_c = t_0
        self.trainer.before_tr()
        for epoch in range(1, self.epochs + 1):
            t_before_epoch = t_c
            flag_stop = self.trainer.tr_epoch(epoch)
            t_c = datetime.datetime.now()
            print(f"epoch: {epoch} ",
                  "now: ", str(t_c),
                  "epoch time: ", t_c - t_before_epoch,
                  "used: ", t_c - t_0,
                  "model: ", self.visitor.model_name)
            # current time, time since experiment start, epoch time
            if flag_stop:
                self.epoch_counter = epoch
                print("early stop trigger")
                break
            if epoch == self.epochs:
                self.epoch_counter = self.epochs
            else:
                self.epoch_counter += 1
        print("Experiment finished at epoch:", self.epoch_counter,
              "with time:", t_c - t_0, "at", t_c)
        self.trainer.post_tr()
