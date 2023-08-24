"""
make an experiment using "jigen" model
"""
from torch import nn
from torchvision import models as torchvisionmodels
from torchvision.models import ResNet50_Weights

from domainlab.mk_exp import mk_exp
from domainlab.dsets.dset_mnist_color_solo_default import DsetMNISTColorSoloDefault
from domainlab.tasks.task_dset import mk_task_dset
from domainlab.models.model_jigen import mk_jigen
from domainlab.tasks.utils_task import ImSize


"""
test mk experiment API with "jigen" model and "mldg" trainer
"""

# specify domain generalization task
task = mk_task_dset(dim_y=10, isize=ImSize(3, 28, 28),  taskna="custom_task")
task.add_domain(name="domain1",
            dset_tr=DsetMNISTColorSoloDefault(0),
            dset_val=DsetMNISTColorSoloDefault(1))
task.add_domain(name="domain2",
            dset_tr=DsetMNISTColorSoloDefault(2),
            dset_val=DsetMNISTColorSoloDefault(3))
task.add_domain(name="domain3",
            dset_tr=DsetMNISTColorSoloDefault(4),
            dset_val=DsetMNISTColorSoloDefault(5))

# specify parameters
num_output_net_classifier = task.dim_y
num_output_net_permutation = 2 #file runs for = 2 but not for anything else (should be 32?)
list_str_y = [f"class{i}" for i in range(num_output_net_classifier)]
list_str_d = ["domain1", "domain2", "domain3"]
coeff_reg = 1e-3

# specify feature extractor as ConvNet
net_encoder = torchvisionmodels.resnet.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_output_net_encoder = net_encoder.fc.out_features

# specify permutation classifier as linear network
net_permutation = nn.Linear(num_output_net_encoder, num_output_net_permutation)

# specify label classifier as linear network
net_classifier = nn.Linear(num_output_net_encoder, num_output_net_classifier)

# specify model to use
model = mk_jigen()(list_str_y, list_str_d, net_encoder, net_classifier, net_permutation, coeff_reg)

# make trainer for model
exp = mk_exp(task, model, trainer="mldg", test_domain="domain1", batchsize=32)
exp.execute(num_epochs=3)
