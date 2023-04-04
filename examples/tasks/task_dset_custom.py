"""
example task from dataset
"""
from torchvision import transforms
from domainlab.tasks.utils_task import ImSize
from domainlab.tasks.task_dset import mk_task_dset
from domainlab.dsets.dset_mnist_color_solo_default import DsetMNISTColorSoloDefault


DICT_DOMAIN2DSET = {}  # list of domains with dset_tr, dset_val
DICT_DOMAIN2DSET["d1"] = (DsetMNISTColorSoloDefault(0, "zout"),
                          DsetMNISTColorSoloDefault(0, "zout"))
DICT_DOMAIN2DSET["d2"] = (DsetMNISTColorSoloDefault(1, "zout"),
                          DsetMNISTColorSoloDefault(1, "zout"))
DICT_DOMAIN2DSET["d3"] = (DsetMNISTColorSoloDefault(2, "zout"),
                          DsetMNISTColorSoloDefault(2, "zout"))


IMG_TRANS = transforms.Compose([transforms.ToTensor()])


chain = mk_task_dset(DICT_DOMAIN2DSET,
                     list_str_y=[str(ele) for ele in range(0, 10)],
                     isize=ImSize(3, 28, 28),
                     dict_domain_img_trans={"d1": IMG_TRANS, "d2": IMG_TRANS, "d3": IMG_TRANS},
                     img_trans_te=IMG_TRANS,
                     taskna="custom_dset")

def get_task(na=None):
    return chain
