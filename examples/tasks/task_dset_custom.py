"""
example task from dataset
"""
import os
from torchvision import transforms
from domainlab.tasks.utils_task import ImSize
from domainlab.tasks.task_dset import mk_task_dset
from domainlab.dsets.dset_mnist_color_solo_default import DsetMNISTColorSoloDefault

# relative path is essential here since this file is used for testing
path_this_file = os.path.dirname(os.path.realpath(__file__))


dict_domain2dset = {}  # list of domains with dset_tr, dset_val
dict_domain2dset["d1"] = (DsetMNISTColorSoloDefault(0, "zout"),
                          DsetMNISTColorSoloDefault(0, "zout"))
dict_domain2dset["d2"] = (DsetMNISTColorSoloDefault(1, "zout"),
                          DsetMNISTColorSoloDefault(1, "zout"))
dict_domain2dset["d3"] = (DsetMNISTColorSoloDefault(2, "zout"),
                          DsetMNISTColorSoloDefault(2, "zout"))

list_str_y = list(range(0, 10))  # list of common class-labels among domains
list_str_y = [str(ele) for ele in list_str_y]

img_trans = transforms.Compose([transforms.ToTensor()])


chain = mk_task_dset(dict_domain2dset=dict_domain2dset,
                     list_str_y=list_str_y,
                     isize=ImSize(3, 28, 28),
                     dict_domain_img_trans={"d1": img_trans, "d2": img_trans, "d3": img_trans},
                     img_trans_te=img_trans,
                     taskna="custom_dset")

def get_task(na=None):
    return chain
