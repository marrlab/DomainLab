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
                      list_str_y=["monocyte", "eosinophil", "basophil"],
                      dict_domain_folder_name2class={
                          "dib": {"monocyte": "monocyte",
                                  # "lymphocyte": "lym",
                                  "basophil": "basophil",
                                  "eosinophil": "eosinophil"},
                          # "neutrophil": "neutrophil"},
                          "mllmar": {"05-MONO": "monocyte",
                                     "10-EOS": "eosinophil",
                                     # "04-LGL": "lym",
                                     # "11-STAB": "neutrophil"
                                     "09-BASO":"basophil"},
                          "amlmatek": {"EOS": "eosinophil",
                                       "MON": "monocyte",
                                       # "LYT": "lym",
                                       # "NGB": "neutrophil"
                                       "BAS": "basophil"}},
                      dict_domain_img_trans={
                          "dib": trans,
                          "mllmar": trans,
                          "amlmatek": trans,
                      },
                      img_trans_te=trans_te,
                      isize=ImSize(3, 224, 224),
                      dict_domain2imgroot={
                          "amlmatek": "/storage/groups/qscd01/datasets/191024_AML_Matek/AML-Cytomorphology_LMU",
                          "mllmar": "/storage/groups/qscd01/datasets/190527_MLL_marr/Data",
                          "dib": "/storage/groups/qscd01/datasets/armingruber/PBC_dataset_normal_DIB/"},
                      taskna="blood_mon_eos_bas")


def get_task(na=None):
    return task
