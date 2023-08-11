"""
make an experiment
"""
from torch import nn
from torchvision import models as torchvisionmodels
from torchvision import transforms
from torchvision.models import ResNet50_Weights

from domainlab.mk_exp import mk_exp
from domainlab.dsets.dset_mnist_color_solo_default import DsetMNISTColorSoloDefault
from domainlab.tasks.task_dset import mk_task_dset
from domainlab.models.model_deep_all import mk_deepall
from domainlab.tasks.utils_task import ImSize


def test_mk_exp():
    # specify domain generalization task
    DICT_DOMAIN2DSET = {}  # build dictionary of domains, here we give domains named "d1", "d2", "d3".
    DICT_DOMAIN2DSET["d1"] = (DsetMNISTColorSoloDefault(0, "zout"),
            DsetMNISTColorSoloDefault(0, "zout"))  # first position in tuple is training, second is validation
    DICT_DOMAIN2DSET["d2"] = (DsetMNISTColorSoloDefault(1, "zout"),  # train and validation for domain "d2"
            DsetMNISTColorSoloDefault(1, "zout"))
    DICT_DOMAIN2DSET["d3"] = (DsetMNISTColorSoloDefault(2, "zout"),
            DsetMNISTColorSoloDefault(2, "zout"))  # train and validation for domain "d3"

    IMG_TRANS = transforms.Compose([transforms.ToTensor()])      # specify transformations to use for training data

    task = mk_task_dset(isize=ImSize(3, 28, 28))

    # specify backbone to use
    dim_y = 10
    backbone = torchvisionmodels.resnet.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_final_in = backbone.fc.in_features
    backbone.fc = nn.Linear(num_final_in, dim_y)

    ## specify model to use
    model = mk_deepall()(backbone)

    ## make trainer for model
    exp = mk_exp(task, model, trainer="mldg", test_domain="d1", batchsize=32)
    exp.execute(num_epochs=1)
