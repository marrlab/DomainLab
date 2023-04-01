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
                self.dataset_sanity_check(args, args.san_num)
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

    def dataset_sanity_check(self, args, sample_num):
        """
        when we load data from folder or a file listing the path of observations,
        we want to check if the file we loaded are in accordance with our expectations
        This function dump a subsample of the dataset into hierarchical folder structure.
        """
        self.task.init_business(args)

        list_domain_tr, list_domain_te = self.task.get_list_domains_tr_te(args.tr_d, args.te_d)


        time_stamp = datetime.datetime.now()
        f_name = os.path.join(args.out, 'Dset_extraction',
                              self.task.task_name + ' ' + str(time_stamp))
        # remove previous sanity checks with the same name
        shutil.rmtree(f_name, ignore_errors=True)

        # for each training domain do...
        for domain in list_domain_tr:
            self.save_san_check_for_domain(args, sample_num, f_name, domain)

        # for each testing domain do...
        for domain in list_domain_te:
            self.save_san_check_for_domain(args, sample_num, f_name, domain, test=True)


    def save_san_check_for_domain(self, args, sample_num, f_name, domain, test=False):
        if not test:
            folder_name = 'train_domain/' + str(domain)
        else:
            folder_name = 'test_domain/' + str(domain)

        d_dataset = self.task.get_dset_by_domain(args, domain)[0]

        # for each class do...
        for class_num in range(len(self.task.list_str_y)):
            num_of_samples = 0
            loader_domain = data_utils.DataLoader(d_dataset, batch_size=1, shuffle=False)
            domain_targets = []
            for num, (_, lab, *_) in enumerate(loader_domain):
                if int(np.argmax(lab[0])) == class_num:
                    domain_targets.append(num)
                    num_of_samples += 1
                if sample_num == num_of_samples:
                    break

            class_dataset = Subset(d_dataset, domain_targets)
            os.makedirs(f_name + '/' + folder_name, exist_ok=True)
            plot_ds(
                class_dataset,
                f_name + '/' + folder_name + '/' +
                str(self.task.list_str_y[class_num]) + '.jpg',
                batchsize=sample_num
            )
