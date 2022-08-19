import os
from torchvision import transforms
from domainlab.tasks.utils_task import ImSize
from domainlab.tasks.task_custom_loader import mk_task_dset
from domainlab.dsets.dset_mnist_color_solo_default import DsetMNISTColorSoloDefault

# relative path is essential here since this file is used for testing, no absolute directory possible

path_this_file = os.path.dirname(os.path.realpath(__file__))

dset_tr = DsetMNISTColorSoloDefault(0, "zout")
dset_val = DsetMNISTColorSoloDefault(0, "zout")
dset_te = DsetMNISTColorSoloDefault(0, "zout")

list_str_y = list(range(0, 10))

list_str_y = [str(ele) for ele in list_str_y]

dict_domain2dset = {}
dict_domain2dset["0"] = (dset_tr, dset_val)
dict_domain2dset["1"] = (dset_tr, dset_val)
dict_domain2dset["2"] = (dset_tr, dset_val)

chain = mk_task_dset(dict_domain2dset=dict_domain2dset,
                      dset_tr=dset_tr,
                      dset_val=dset_val,
                      dset_te=dset_te,
                      list_str_y=list_str_y,
                      isize=ImSize(3, 28, 28),
                      taskna="custom_dset")


def get_task(na=None):
    return chain
