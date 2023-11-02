"""
test task for image size 224
"""
import os

from torchvision import transforms

from domainlab.tasks.task_folder_mk import mk_task_folder
from domainlab.tasks.utils_task import ImSize

path_this_file = os.path.dirname(os.path.realpath(__file__))


def addtask2chain(chain):
    """
    given a chain of responsibility for task selection, add another task into the chain
    """
    new_chain = mk_task_folder(extensions={"caltech": "jpg", "sun": "jpg", "labelme": "jpg"},
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
                                   "caltech": os.path.join(
                                       path_this_file,
                                       "../../data/vlcs_mini/caltech/"),
                                   "sun": os.path.join(
                                       path_this_file,
                                       "../../data/vlcs_mini/sun/"),
                                   "labelme": os.path.join(
                                       path_this_file,
                                       "../../data/vlcs_mini/labelme/")},
                               taskna="mini_vlcs",
                               succ=chain)
    return new_chain
