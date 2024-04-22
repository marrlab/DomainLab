"""
full data(images), as well as the txt file indicating the filenames of images can be
download from
- (https://domaingeneralization.github.io/#data)
- or (https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk?resourcekey=0-2fvpQY_QSyJf2uIECzqPuQ)
"""

from torchvision import transforms

from domainlab.tasks.task_pathlist import mk_node_task_path_list
from domainlab.tasks.utils_task import ImSize

# change this to absolute directory where you have the raw images from PACS,
G_PACS_RAW_PATH = "domainlab/zdata/pacs/PACS"
# domainlab repository contain already the file names in domainlab/zdata/pacs_split folder of domainlab


def get_task(na=None):
    node = mk_node_task_path_list(
        isize=ImSize(3, 224, 224),
        list_str_y=["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"],
        dict_class_label_ind2name={
            "1": "dog",
            "2": "elephant",
            "3": "giraffe",
            "4": "guitar",
            "5": "horse",
            "6": "house",
            "7": "person",
        },
        dict_d2filepath_list_img_tr={
            "art_painting": "domainlab/zdata/pacs_split/art_painting_train_kfold.txt",
            "cartoon": "domainlab/zdata/pacs_split/cartoon_train_kfold.txt",
            "photo": "domainlab/zdata/pacs_split/photo_train_kfold.txt",
            "sketch": "domainlab/zdata/pacs_split/sketch_train_kfold.txt",
        },
        dict_d2filepath_list_img_te={
            "art_painting": "domainlab/zdata/pacs_split/art_painting_test_kfold.txt",
            "cartoon": "domainlab/zdata/pacs_split/cartoon_test_kfold.txt",
            "photo": "domainlab/zdata/pacs_split/photo_test_kfold.txt",
            "sketch": "domainlab/zdata/pacs_split/sketch_test_kfold.txt",
        },
        dict_d2filepath_list_img_val={
            "art_painting": "domainlab/zdata/pacs_split/art_painting_crossval_kfold.txt",
            "cartoon": "domainlab/zdata/pacs_split/cartoon_crossval_kfold.txt",
            "photo": "domainlab/zdata/pacs_split/photo_crossval_kfold.txt",
            "sketch": "domainlab/zdata/pacs_split/sketch_crossval_kfold.txt",
        },
        dict_domain2imgroot={
            "art_painting": G_PACS_RAW_PATH,
            "cartoon": G_PACS_RAW_PATH,
            "photo": G_PACS_RAW_PATH,
            "sketch": G_PACS_RAW_PATH,
        },
        img_trans_tr=transforms.Compose(
            [

                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        img_trans_te=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
    return node
