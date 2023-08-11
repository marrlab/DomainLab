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

    task = mk_task_dset(isize=ImSize(3, 28, 28),  taskna="custom_task")
    task.add_domain(name="domain1",
                    dset_tr=DsetMNISTColorSoloDefault(0, "zoutput"),
                    dset_val=DsetMNISTColorSoloDefault(1, "zoutput"))
    task.add_domain(name="domain2",
                    dset_tr=DsetMNISTColorSoloDefault(2, "zoutput"),
                    dset_val=DsetMNISTColorSoloDefault(3, "zoutput"))
    task.add_domain(name="domain3",
                    dset_tr=DsetMNISTColorSoloDefault(4, "zoutput"),
                    dset_val=DsetMNISTColorSoloDefault(5, "zoutput"))

    # specify backbone to use
    dim_y = 10
    backbone = torchvisionmodels.resnet.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_final_in = backbone.fc.in_features
    backbone.fc = nn.Linear(num_final_in, dim_y)

    # specify model to use
    model = mk_deepall()(backbone)

    # make trainer for model
    exp = mk_exp(task, model, trainer="mldg", test_domain="domain1", batchsize=32)
    exp.execute(num_epochs=1)
