from torchvision import transforms
from libdg.tasks.task_folder_mk import mk_task_folder
from libdg.tasks.utils_task import ImSize


task = mk_task_folder(extensions={"dib": "jpg", "mllmar": "TIF", "amlmatek": "tiff"},
                      list_str_y=["monocyte", "eosinophil", "basophil"],
                      dict_domain_folder_name2class={
                          "dib": {"monocyte": "monocyte",
                                  # "lymphocyte": "lym",
                                  "basophil": "basophil",
                                  "eosinophil": "eosinophil"},
                          # "neutrophil": "neutrophil"},
                          "mllmar": {"05MONO": "monocyte",
                                     "10EOS": "eosinophil",
                                     # "04-LGL": "lym",
                                     # "11-STAB": "neutrophil"
                                     "09-BASO":"basophil"},
                          "amlmatek": {"EOS": "eosinophil",
                                       "MON": "monocyte",
                                       # "LYT": "lym",
                                       # "NGB": "neutrophil"
                                       "BAS": "basophil"}},
                      dict_domain_img_trans={
                          "dib": transforms.Compose([transforms.Resize((224, 224)), ]),
                          "mllmar": transforms.Compose([transforms.Resize((224, 224)), ]),
                          "amlmatek": transforms.Compose([transforms.Resize((224, 224)), ]),
                      },
                      isize=ImSize(3, 224, 224),
                      dict_domain2imgroot={
                          "amlmatek": "/storage/groups/qscd01/datasets/191024_AML_Matek/AML-Cytomorphology_LMU",
                          "mllmar": "/storage/groups/qscd01/datasets/190527_MLL_marr/Data",
                          "dib": "/storage/groups/qscd01/datasets/armingruber/PBC_dataset_normal_DIB/"},
                      taskna="blood_mon_eos_bas")


def get_task(na=None):
    return task
