import os

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


def test_dset_sample_extraction():
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

 parser = mk_parser_main()
 args = parser.parse_args(["--te_d", "1", "--bs", "2", "--aname", "diva"])
 task.init_business(args)
 task.get_list_domains()

 dset_name = 'vlcs_mini'
 if not os.path.exists('zout/Dset_extraction/'):
    os.mkdir('zout/Dset_extraction/')
 f_name = 'zout/Dset_extraction/' + dset_name
 sample_num = 8
 if not os.path.exists(f_name):
    os.mkdir(f_name)

 # for each domain do...
 for domain in task.get_list_domains():
    # generate a dataset for each domain
    d_dataset = task.get_dset_by_domain(args, domain)[0]

    if not os.path.exists(f_name + '/' + str(domain)):
        os.mkdir(f_name + '/' + str(domain))

    # for each class do...
    for class_num in range(len(d_dataset.dset.classes)):
        # find indices corresponding to one class
        domain_targets = np.where(np.array(d_dataset.targets) == class_num)
        # create a dataset subset containing only images of one class
        class_dataset = Subset(d_dataset, domain_targets[0])
        # plot the images of this class and save it with its specific file name
        full_f_name = f_name + '/' + str(domain) + '/' + str(d_dataset.dict_folder_name2class_global[d_dataset.dset.classes[class_num]]) + '.jpg'
        plot_ds(class_dataset, full_f_name, bs=sample_num)
