"""
difference than task_blood2.py:

"mll": "/lustre/groups/labs/marr/qscd01/datasets/240416_MLL23",

"""
from torchvision import transforms

from domainlab.tasks.task_folder_mk import mk_task_folder
from domainlab.tasks.utils_task import ImSize

IMG_SIZE = 224

trans = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

trans_te = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


TASK = mk_task_folder(
    extensions={"acevedo": "jpg", "matek": "tiff", "mll": "tif"},
    list_str_y=[
        "basophil",
        "erythroblast",
        "metamyelocyte",
        "myeloblast",
        "neutrophil_band",
        "promyelocyte",
        "eosinophil",
        "lymphocyte_typical",
        "monocyte",
        "myelocyte",
        "neutrophil_segmented",
    ],
    dict_domain_folder_name2class={
        "acevedo": {
            "basophil": "basophil",
            "erythroblast": "erythroblast",
            "metamyelocyte": "metamyelocyte",
            "neutrophil_band": "neutrophil_band",
            "promyelocyte": "promyelocyte",
            "eosinophil": "eosinophil",
            "lymphocyte_typical": "lymphocyte_typical",
            "monocyte": "monocyte",
            "myelocyte": "myelocyte",
            "neutrophil_segmented": "neutrophil_segmented",
        },
        "matek": {
            "basophil": "basophil",
            "erythroblast": "erythroblast",
            "metamyelocyte": "metamyelocyte",
            "myeloblast": "myeloblast",
            "neutrophil_band": "neutrophil_band",
            "promyelocyte": "promyelocyte",
            "eosinophil": "eosinophil",
            "lymphocyte_typical": "lymphocyte_typical",
            "monocyte": "monocyte",
            "myelocyte": "myelocyte",
            "neutrophil_segmented": "neutrophil_segmented",
        },
        "mll": {
            "basophil": "basophil",
            "normoblast": "normoblast",
            "metamyelocyte": "metamyelocyte",
            "myeloblast": "myeloblast",
            "neutrophil_band": "neutrophil_band",
            "promyelocyte": "promyelocyte",
            "eosinophil": "eosinophil",
            "lymphocyte": "lymphocyte",
            "monocyte": "monocyte",
            "myelocyte": "myelocyte",
            "neutrophil_segmented": "neutrophil_segmented",
        },
    },
    dict_domain_img_trans={
        "acevedo": trans,
        "mll": trans,
        "matek": trans,
    },
    img_trans_te=trans_te,
    isize=ImSize(3, IMG_SIZE, IMG_SIZE),
    dict_domain2imgroot={
        "matek": "/lustre/groups/labs/marr/qscd01/datasets/armingruber/_Domains/Matek_cropped",
        "mll": "/lustre/groups/labs/marr/qscd01/datasets/240416_MLL23",
        "acevedo": "/lustre/groups/labs/marr/qscd01/datasets/armingruber/_Domains/Acevedo_cropped",
    },
    taskna="blood_mon_eos_bas",
)


def get_task(na=None):
    return TASK
