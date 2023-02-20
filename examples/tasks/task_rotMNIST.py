import os
from torchvision import transforms
from domainlab.tasks.task_folder_mk import mk_task_folder
from domainlab.tasks.utils_task import ImSize

# relative path is essential here since this file is used for testing, no absolute directory possible

path_this_file = os.path.dirname(os.path.realpath(__file__))
chain = mk_task_folder(extensions=None,
                       list_str_y=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                       dict_domain_folder_name2class={
                           "0deg": {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
                                    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9"}
                       },
                       dict_domain_img_trans={
                           "0deg": transforms.Compose(
                               [transforms.Resize((28, 28)),
                                transforms.ToTensor(),
                                transforms.RandomRotation((0, 60))
                                ])
                       },
                       img_trans_te=transforms.Compose(
                           [transforms.Resize((28, 28)),
                            transforms.ToTensor(),
                            transforms.RandomRotation((90, 100))
                            ]),
                       isize=ImSize(3, 28, 28),
                       dict_domain2imgroot={
                           "caltech": os.path.join(
                               path_this_file,
                               "../../data/MNIST/0deg/")
                       },
                       taskna="rotatedMNIST")


def get_task(na=None):
    return chain
