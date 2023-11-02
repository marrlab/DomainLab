'''
This class is used to perform the sanity check on a task description
'''

import datetime
import os
import shutil
import numpy as np
from torch.utils.data import Subset
import torch.utils.data as data_utils

from domainlab.dsets.utils_data import plot_ds


class SanityCheck():
    """
    Performs a sanity check on the given args and the task
    when running dataset_sanity_check(self)
    """
    def __init__(self, args, task):
        self.args = args
        self.task = task

    def dataset_sanity_check(self):
        """
        when we load data from folder or a file listing the path of observations,
        we want to check if the file we loaded are in accordance with our expectations
        This function dump a subsample of the dataset into hierarchical folder structure.
        """
        # self.task.init_business(self.args)

        list_domain_tr, list_domain_te = self.task.get_list_domains_tr_te(self.args.tr_d,
                                                                          self.args.te_d)


        time_stamp = datetime.datetime.now()
        f_name = os.path.join(self.args.out, 'Dset_extraction',
                              self.task.task_name + ' ' + str(time_stamp))
        # remove previous sanity checks with the same name
        shutil.rmtree(f_name, ignore_errors=True)

        # for each training domain do...
        for domain in list_domain_tr:
            if domain in self.task.dict_dset_tr:
                d_dataset = self.task.dict_dset_tr[domain]
            else:
                d_dataset = self.task.get_dset_by_domain(self.args, domain)[0]
            folder_name = f_name + '/train_domain/' + str(domain)
            self.save_san_check_for_domain(self.args.san_num, folder_name, d_dataset)

        # for each testing domain do...
        for domain in list_domain_te:
            if domain in self.task.dict_dset_te:
                d_dataset = self.task.dict_dset_te[domain]
            else:
                d_dataset = self.task.get_dset_by_domain(self.args, domain)[0]
            folder_name = f_name + '/test_domain/' + str(domain)
            self.save_san_check_for_domain(self.args.san_num, folder_name, d_dataset)


    def save_san_check_for_domain(self, sample_num, folder_name, d_dataset):
        '''
        saves a extraction of the dataset (d_dataset) into folder (folder_name)
        sample_num: int, number of images which are extracted from the dataset
        folder_name: string, destination for the saved images
        d_dataset: dataset
        '''

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
            os.makedirs(folder_name, exist_ok=True)
            plot_ds(
                class_dataset,
                folder_name + '/' +
                str(self.task.list_str_y[class_num]) + '.jpg',
                batchsize=sample_num
            )
