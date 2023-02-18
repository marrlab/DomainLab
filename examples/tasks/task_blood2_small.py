from torchvision import transforms
from domainlab.tasks.task_folder_mk import mk_task_folder
from domainlab.tasks.utils_task import ImSize


img_size = 48
# img_size = 28

trans = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

trans_te = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


task = mk_task_folder(extensions={"acevedo": "jpg", "matek": "tiff", "mll": "tif"},
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
                          "neutrophil_segmented"
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
                      },
                      dict_domain_img_trans={
                          "acevedo": trans,
                          "mll": trans,
                          "matek": trans,
                      },
                      img_trans_te=trans_te,
                      isize=ImSize(3, img_size, img_size),
                      dict_domain2imgroot={
                          "matek": "/home/ubuntu/_Domains/Matek/",
                          "mll": "/home/ubuntu/_Domains/MLL/",
                          "acevedo": "/home/ubuntu/_Domains/Acevedo/"},
                      taskna="blood_mon_eos_bas")


def get_task(na=None):
    return task
