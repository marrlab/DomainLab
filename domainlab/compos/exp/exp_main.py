import datetime
import os
from torch.utils.data import Subset
import torch.utils.data as data_utils
import numpy as np

from domainlab.algos.zoo_algos import AlgoBuilderChainNodeGetter
from domainlab.compos.exp.exp_utils import AggWriter
from domainlab.tasks.zoo_tasks import TaskChainNodeGetter
from domainlab.dsets.utils_data import plot_ds

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # debug


class Exp():
    """
    Exp is combination of Task, Algorithm, and Configuration (including random seed)
    """
    def __init__(self, args, task=None):
        """
        :param args:
        :param task:
        """
        self.task = task
        if task is None:
            self.task = TaskChainNodeGetter(args)()
            if args.san_check:
                self.dataset_sanity_check(args, args.san_num)
        self.task.init_business(args)
        self.args = args
        self.visitor = AggWriter(self)
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
            print("now: ", str(t_c), "epoch time: ", t_c - t_before_epoch, "used: ", t_c - t_0)
            # current time, time since experiment start, epoch time
            if flag_stop:
                self.epoch_counter = epoch
                break
            elif epoch == self.epochs:
                self.epoch_counter = self.epochs
            else:
                self.epoch_counter += 1
        print("Experiment finished at epoch:", self.epoch_counter,
              "with time:", t_c - t_0, "at", t_c)
        self.trainer.post_tr()

    def dataset_sanity_check(self, args, sample_num):
        self.task.init_business(args)

        dset_name = self.task.task_name
        if not os.path.exists('zoutput/Dset_extraction/'):
            os.mkdir('zoutput/Dset_extraction/')
        f_name = 'zoutput/Dset_extraction/' + dset_name
        if not os.path.exists(f_name):
            os.mkdir(f_name)

        # for each domain do...
        for domain in self.task.get_list_domains():
            # generate a dataset for each domain
            d_dataset = self.task.get_dset_by_domain(args, domain)[0]

            if not os.path.exists(f_name + '/' + str(domain)):
                os.mkdir(f_name + '/' + str(domain))

            # for each class do...
            for class_num in range(len(self.task.list_str_y)):
                num_of_samples = 0
                loader_domain = data_utils.DataLoader(d_dataset, batch_size=1, shuffle=False)
                domain_targets = []
                image_list = []
                label_list = []
                for num, (img, lab, *_) in enumerate(loader_domain):
                    if int(np.argmax(lab[0])) == class_num:
                        domain_targets.append(num)
                        num_of_samples += 1
                        img_ = np.moveaxis(np.array(img[0]), 0, -1)
                        image_list.append(img_)
                        label_list.append(lab)
                    if sample_num == num_of_samples:
                        break

                class_dataset = Subset(d_dataset, domain_targets)
                full_f_name = f_name + '/' + str(domain) + '/' + str(self.task.list_str_y[class_num]) + '.jpg'
                plot_ds(class_dataset, full_f_name, bs=sample_num)