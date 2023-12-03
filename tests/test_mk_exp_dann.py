"""
make an experiment using "dann" model
"""
from torch import nn
from torchvision import models as torchvisionmodels
from torchvision.models import ResNet50_Weights

from domainlab.mk_exp import mk_exp
from domainlab.dsets.dset_mnist_color_solo_default import DsetMNISTColorSoloDefault
from domainlab.tasks.task_dset import mk_task_dset
from domainlab.models.model_dann import mk_dann
from domainlab.tasks.utils_task import ImSize


def test_mk_exp_dann():
    """
    test mk experiment API with "dann" model and "mldg", "dial" trainers
    """
    mk_exp_dann(trainer="basic")
    mk_exp_dann(trainer="mldg")
    mk_exp_dann(trainer="dial")


def mk_exp_dann(trainer="mldg"):
    """
    execute experiment with "dann" model and arbitrary trainer
    """

    # specify domain generalization task
    task = mk_task_dset(dim_y=10, isize=ImSize(3, 28, 28), taskna="custom_task")
    task.add_domain(name="domain1",
                    dset_tr=DsetMNISTColorSoloDefault(0),
                    dset_val=DsetMNISTColorSoloDefault(1))
    task.add_domain(name="domain2",
                    dset_tr=DsetMNISTColorSoloDefault(2),
                    dset_val=DsetMNISTColorSoloDefault(3))
    task.add_domain(name="domain3",
                    dset_tr=DsetMNISTColorSoloDefault(4),
                    dset_val=DsetMNISTColorSoloDefault(5))
    task.get_list_domains_tr_te(None, "domain1")
    # specify task-specific parameters
    num_output_net_classifier = task.dim_y
    num_output_net_discriminator = len(task.list_domain_tr)
    list_str_y = [f"class{i}" for i in range(num_output_net_classifier)]
    alpha = 1e-3

    # specify feature extractor as ResNet
    net_encoder = torchvisionmodels.resnet.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_output_net_encoder = net_encoder.fc.out_features

    # specify discriminator as linear network
    net_discriminator = nn.Linear(num_output_net_encoder, num_output_net_discriminator)

    # specify net classifier as linear network
    net_classifier = nn.Linear(num_output_net_encoder, num_output_net_classifier)

    # specify model to use
    model = mk_dann()(list_str_y, task.list_domain_tr, alpha, net_encoder, net_classifier, net_discriminator)

    # make trainer for model
    
    exp = mk_exp(task, model, trainer=trainer, test_domain="domain1", batchsize=32)
    exp.execute(num_epochs=2)
