from torchvision import transforms
from domainlab.tasks.task_folder_mk import mk_task_folder
from domainlab.tasks.utils_task import ImSize



trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

trans_te = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


task = mk_task_folder(extensions={"dib": "jpg", "mllmar": "TIF", "amlmatek": "tiff"},

                    list_str_y=["basophil", "eosinophil", "erythroblast", "lymphocyte_typical", "metamyelocyte", "monocyte", "myeloblast", "myelocyte", "neutrophil_banded", "neutrophil_segmented", "promyelocyte"],


                      dict_domain_folder_name2class={
                          "dib" : {"basophil": "basophil",
                                    "eosinophil": "eosinophil",
                                    "erythroblast": "erythroblast",
                                    "lymphocyte_typical": "lymphocyte_typical",
                                    "metamyelocyte": "metamyelocyte",
                                    "monocyte": "monocyte",
                                    "promyelocyte": "promyelocyte",
                                    "myelocyte": "myelocyte",
                                    "neutrophil_banded": "neutrophil_banded",
                                    "neutrophil_segmented": "neutrophil_segmented", 
                                    },
                          
                          "mllmar" : {"basophil": "basophil",
                                    "eosinophil": "eosinophil",
                                    "erythroblast": "erythroblast",
                                    "myeloblast": "myeloblast",
                                    "promyelocyte": "promyelocyte",
                                    "myelocyte": "myelocyte",
                                    "metamyelocyte": "metamyelocyte",
                                    "neutrophil_banded": "neutrophil_banded",
                                    "neutrophil_segmented": "neutrophil_segmented",
                                    "monocyte": "monocyte",
                                    "lymphocyte_typical": "lymphocyte_typical",
                                    },
                          
                          "amlmatek" : {"basophil": "basophil",
                                    "eosinophil": "eosinophil",
                                    "erythroblast": "erythroblast",
                                    "myeloblast": "myeloblast",
                                    "promyelocyte": "promyelocyte",
                                    "myelocyte": "myelocyte",
                                    "metamyelocyte": "metamyelocyte",
                                    "neutrophil_banded": "neutrophil_banded",
                                    "neutrophil_segmented": "neutrophil_segmented",
                                    "monocyte": "monocyte",
                                    "lymphocyte_typical": "lymphocyte_typical",
                                    }},

                    dict_domain_img_trans={
                          "dib": trans,
                          "mllmar": trans,
                          "amlmatek": trans,
                      },
                      img_trans_te=trans_te,
                      isize=ImSize(3, 224, 224),
                      dict_domain2imgroot={
                          "amlmatek": "/lustre/groups/labs/marr/qscd01/datasets/armingruber/_datasets/Mat19_cropped",
                          "mllmar": "/lustre/groups/labs/marr/qscd01/datasets/armingruber/_datasets/_Test",
                          "dib": "/lustre/groups/labs/marr/qscd01/datasets/armingruber/_datasets/Ace20_cropped"},
                      taskna="blood_hack")


def get_task(na=None):
    return task
