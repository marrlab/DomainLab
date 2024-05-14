"""
Toy example on how to use TaskPathList, by subsample a small portion of PACS data
"""
from torchvision import transforms

from domainlab.tasks.task_pathlist import mk_node_task_path_list
from domainlab.tasks.utils_task import ImSize


def get_task(na=None):
    """
    specify task by providing list of training, test and validation paths
    folder path is with respect to the current working directory
    """
    node = mk_node_task_path_list(
        # ## specify image size, must be consistent with the transformation
        isize=ImSize(3, 224, 224),
        # ## specify the names for all classes to classify
        list_str_y=["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"],
        # ## give an index to each target class
        dict_class_label_ind2name={
            "1": "dog",
            "2": "elephant",
            "3": "giraffe",
            "4": "guitar",
            "5": "horse",
            "6": "house",
            "7": "person",
        },
        # ## specify where to find the text file containing path for each image
        # ## the text file below only need to specify the relative path
        # ## training
        dict_d2filepath_list_img_tr={
            "art_painting": "domainlab/zdata/pacs_split/art_painting_10.txt",
            "cartoon": "domainlab/zdata/pacs_split/cartoon_10.txt",
            "photo": "domainlab/zdata/pacs_split/photo_10.txt",
            "sketch": "domainlab/zdata/pacs_split/sketch_10.txt",
        },
        # ## testing
        dict_d2filepath_list_img_te={
            "art_painting": "domainlab/zdata/pacs_split/art_painting_10.txt",
            "cartoon": "domainlab/zdata/pacs_split/cartoon_10.txt",
            "photo": "domainlab/zdata/pacs_split/photo_10.txt",
            "sketch": "domainlab/zdata/pacs_split/sketch_10.txt",
        },
        # ## validation
        dict_d2filepath_list_img_val={
            "art_painting": "domainlab/zdata/pacs_split/art_painting_10.txt",
            "cartoon": "domainlab/zdata/pacs_split/cartoon_10.txt",
            "photo": "domainlab/zdata/pacs_split/photo_10.txt",
            "sketch": "domainlab/zdata/pacs_split/sketch_10.txt",
        },
        # ## specify root folder storing the images of each domain:
        dict_domain2imgroot={
            "art_painting": "domainlab/zdata/pacs_mini_10",
            "cartoon": "domainlab/zdata/pacs_mini_10",
            "photo": "domainlab/zdata/pacs_mini_10",
            "sketch": "domainlab/zdata/pacs_mini_10",
        },
        # ## specify the pytorch transformation you want to apply to the image
        img_trans_tr=transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        ),
        img_trans_te=transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        ),
    )
    return node
