import os
from torchvision import transforms
from domainlab.tasks.task_folder_mk import mk_task_folder
from domainlab.tasks.utils_task import ImSize

# relative path is essential here since this file is used for testing, no absolute directory possible

path_this_file = os.path.dirname(os.path.realpath(__file__))
chain = mk_task_folder(extensions={"caltech": "jpg", "sun":
                                   "jpg", "sun": "jpg"},
                       list_str_y=["chair", "car"],
                       dict_domain_folder_name2class={
                           "caltech": {"auto": "car", "stuhl": "chair"},
                           "sun": {"vehicle": "car", "sofa": "chair"},
                           "sun": {"drive": "car", "sit": "chair"}
                       },
                       dict_domain_img_trans={
                           "caltech": transforms.Compose(
                               [transforms.Resize((224, 224)),
                                transforms.ToTensor()]),
                           "sun": transforms.Compose(
                               [transforms.Resize((224, 224)),
                                transforms.ToTensor()]),
                           "sun": transforms.Compose(
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
                           "sun": os.path.join(
                               path_this_file,
                               "../../data/vlcs_mini/labelme/")},
                       taskna="e_mini_vlcs")


def get_task(na=None):
    return chain
