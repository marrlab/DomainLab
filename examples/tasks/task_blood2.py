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


task = mk_task_folder(extensions={"acevedo": "jpg", "matek": "tiff"},
                      list_str_y=["monocyte", "eosinophil"],
                      dict_domain_folder_name2class={
                          "acevedo": {"monocyte": "monocyte",
                                  "eosinophil": "eosinophil"},
                          "matek": {"eosinophil": "eosinophil",
                                       "monocyte": "monocyte"}},
                      dict_domain_img_trans={
                          "acevedo": trans,
                          #"mll": trans,
                          "matek": trans,
                      },
                      img_trans_te=trans_te,
                      isize=ImSize(3, 224, 224),
                      dict_domain2imgroot={
                          "matek": "/home/ubuntu/_Domains/Matek_cropped/",
                          #"mll": "/home/ubuntu/_Domains/MLL/",
                          "acevedo": "/home/ubuntu/_Domains/Acevedo_cropped/"},
                      taskna="blood_mon_eos_bas")


def get_task(na=None):
    return task
