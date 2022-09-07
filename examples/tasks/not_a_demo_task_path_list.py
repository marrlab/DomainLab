"""
full data(images), as well as the txt file indicating the filenames of images can be
download from
- (https://domaingeneralization.github.io/#data)
- or (https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk?resourcekey=0-2fvpQY_QSyJf2uIECzqPuQ)
"""

from torchvision import transforms
from domainlab.tasks.task_pathlist import mk_node_task_path_list
from domainlab.tasks.utils_task import ImSize


def get_task(na=None):
    node = mk_node_task_path_list(
        isize=ImSize(3, 224, 224),
        list_str_y=["dog", "elephant", "giraffe", "guitar",
                    "horse", "house", "person"],
        dict_class_label2name={"1": "dog",
                               "2": "elephant",
                               "3": "giraffe",
                               "4": "guitar",
                               "5": "horse",
                               "6": "house",
                               "7": "person"},
        dict_d2filepath_list_img={
            "art_painting": "~/Documents/datasets/pacs_split/art_painting_train_kfold.txt",
            "cartoon": "~/Documents/datasets/pacs_split/cartoon_train_kfold.txt",
            "photo": "~/Documents/datasets/pacs_split/photo_train_kfold.txt",
            "sketch": "~/Documents/datasets/pacs_split/sketch_train_kfold.txt"},

        dict_d2filepath_list_img_te={
            "art_painting": "~/Documents/datasets/pacs_split/art_painting_train_kfold.txt",
            "cartoon": "~/Documents/datasets/pacs_split/cartoon_train_kfold.txt",
            "photo": "~/Documents/datasets/pacs_split/photo_train_kfold.txt",
            "sketch": "~/Documents/datasets/pacs_split/sketch_train_kfold.txt"},

        dict_d2filepath_list_img_val={
            "art_painting": "~/Documents/datasets/pacs_split/art_painting_crossval_kfold.txt",
            "cartoon": "~/Documents/datasets/pacs_split/cartoon_train_kfold.txt",
            "photo": "~/Documents/datasets/pacs_split/photo_train_kfold.txt",
            "sketch": "~/Documents/datasets/pacs_split/sketch_train_kfold.txt"},

        dict_domain2imgroot={
            'art_painting': "~/Documents/datasets/pacs/raw",
            'cartoon': "~/Documents/datasets/pacs/raw",
            'photo': "~/Documents/datasets/pacs/raw",
            'sketch': "~/Documents/datasets/pacs/raw"},
        trans4all=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))  # FIXME:
    #  FIXME: 224 is already provided in image size
    return node
