"""
example task on how to specify the task for DomainLab directly but specifying a pair of datasets (training and validation (can be None) ) for each domain.
Follow the example to construct your own task to feed into DomainLab. 
"""
from torchvision import transforms
from domainlab.tasks.utils_task import ImSize
from domainlab.tasks.task_dset import mk_task_dset
from domainlab.dsets.dset_mnist_color_solo_default import DsetMNISTColorSoloDefault


DICT_DOMAIN2DSET = {}  # build dictionary of domains, here we give domains named "d1", "d2", "d3".
DICT_DOMAIN2DSET["d1"] = (DsetMNISTColorSoloDefault(0, "zout"),
                          DsetMNISTColorSoloDefault(0, "zout"))  # first position in tuple is training, second is validation
DICT_DOMAIN2DSET["d2"] = (DsetMNISTColorSoloDefault(1, "zout"),  # train and validation for domain "d2"
                          DsetMNISTColorSoloDefault(1, "zout"))
DICT_DOMAIN2DSET["d3"] = (DsetMNISTColorSoloDefault(2, "zout"), 
                          DsetMNISTColorSoloDefault(2, "zout"))  # train and validation for domain "d3"


IMG_TRANS = transforms.Compose([transforms.ToTensor()])      # specify transformations to use for training data


chain = mk_task_dset(DICT_DOMAIN2DSET,  # the constructed dictionary above
                     list_str_y=[str(ele) for ele in range(0, 10)],  # a list of strings naming each class to classify
                     isize=ImSize(3, 28, 28),   # the image size
                     dict_domain_img_trans={"d1": IMG_TRANS, "d2": IMG_TRANS, "d3": IMG_TRANS},  # for each domain, one could specify different transformations
                     img_trans_te=IMG_TRANS, # use the same transformation for test set
                     taskna="custom_dset")  # give a name for your task constructed.

def get_task(na=None):
    return chain
