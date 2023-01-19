import os

import matplotlib.pyplot as plt

from domainlab.dsets.dset_mnist_color_solo_default import DsetMNISTColorSoloDefault
import torch.utils.data as data_utils

from domainlab.dsets.utils_data import plot_ds
from domainlab.tasks.task_folder_mk import mk_task_folder
from torchvision import transforms
from domainlab.tasks.utils_task import ImSize
from domainlab.arg_parser import mk_parser_main
from domainlab.tasks.zoo_tasks import TaskChainNodeGetter
from torchvision.utils import save_image
import numpy as np
from torch.utils.data import Subset
from domainlab.tasks.task_mnist_color import NodeTaskMNISTColor10


def test_dset_sample_extraction(mode='MNIST', show_plot=True):
 # for mode == 'MNIST' is uses the MINISTcolor10 dataset, othervise the vlcs_mini dataset is used
 # show_plot == True makes plot appear for the dataset for each domain and each class (one at a time)
 # results are saved in 'zout/Dset_extraction/...'

 parser = mk_parser_main()

 if not mode == 'MNIST':
     # vlcs_mini task
     task = mk_task_folder(extensions={"caltech": "jpg", "sun": "jpg", "labelme": "jpg"},
                              list_str_y=["chair", "car"],
                                dict_domain_folder_name2class={
                                    "caltech": {"auto": "car",
                                                "stuhl": "chair"},
                                    "sun": {"vehicle": "car",
                                            "sofa": "chair"},
                                    "labelme": {"drive": "car",
                                                "sit": "chair"}
                                },
                                dict_domain_img_trans={
                                    "caltech": transforms.Compose(
                                        [transforms.Resize((224, 224)),
                                        transforms.ToTensor()]),
                                    "sun": transforms.Compose(
                                        [transforms.Resize((224, 224)),
                                        transforms.ToTensor()]),
                                    "labelme": transforms.Compose(
                                        [transforms.Resize((224, 224)),
                                        transforms.ToTensor()]),
                                },
                                img_trans_te=transforms.Compose(
                                    [transforms.Resize((224, 224)),
                                    transforms.ToTensor()]),
                                isize=ImSize(3, 224, 224),
                                dict_domain2imgroot={
                                    "caltech":
                                        "../data/vlcs_mini/caltech/",
                                    "sun":
                                        "../data/vlcs_mini/sun/",
                                    "labelme":
                                        "../data/vlcs_mini/labelme/"},
                                taskna="mini_vlcs",
                                succ=None)

     # vlcs_mini args
     args = parser.parse_args(["--te_d", "1", "--bs", "2", "--aname", "diva", '--tpath', 'examples/tasks/task_vlcs.py', '--nname', 'conv_bn_pool_2'])
     dset_name = 'vlcs_mini'
 else:
     # MNISTColor10 task
     task = NodeTaskMNISTColor10(None)

     # MNISTColor10 args
     args = parser.parse_args(["--te_d", "1", "--bs", "2", "--aname", "diva", '--task', 'mnistcolor10', '--nname', 'conv_bn_pool_2'])
     dset_name = 'MNISTColor10'

 task.init_business(args)

 if not os.path.exists('zout/Dset_extraction/'):
    os.mkdir('zout/Dset_extraction/')
 f_name = 'zout/Dset_extraction/' + dset_name
 sample_num = 4
 if not os.path.exists(f_name):
    os.mkdir(f_name)

 # for each domain do...
 for domain in task.get_list_domains():
    # generate a dataset for each domain
    d_dataset = task.get_dset_by_domain(args, domain)[0]

    if not os.path.exists(f_name + '/' + str(domain)):
        os.mkdir(f_name + '/' + str(domain))

    # for each class do...
    #for class_num in range(len(d_dataset.dset.classes)):
    for class_num in range(len(task.list_str_y)):
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
                #plt.figure()
                #plt.imshow(img_)
                #plt.title(str(lab))
                #plt.show()
            if sample_num == num_of_samples:
                break

        if show_plot:
            fig, ax = plt.subplots(sample_num)
            for i in range(sample_num):
                ax[i].imshow(np.array(image_list[i]))
                ax[i].set_title(str(np.array(label_list[i])) + ', label: ' + str(task.list_str_y[np.argmax(np.array(label_list[i]))]))
            plt.tight_layout()
            plt.show()

        class_dataset = Subset(d_dataset, domain_targets)
        full_f_name = f_name + '/' + str(domain) + '/' + str(task.list_str_y[class_num]) + '.jpg'
        plot_ds(class_dataset, full_f_name, bs=sample_num)

