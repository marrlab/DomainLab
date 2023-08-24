"""
make an experiment using "diva" model
"""

from domainlab.mk_exp import mk_exp
from domainlab.dsets.dset_mnist_color_solo_default import DsetMNISTColorSoloDefault
from domainlab.tasks.task_dset import mk_task_dset
from domainlab.models.model_diva import mk_diva
from domainlab.tasks.utils_task import ImSize
from domainlab.compos.vae.a_vae_builder import AbstractVAEBuilderChainNode
from domainlab.compos.vae.c_vae_builder_classif import ChainNodeVAEBuilderClassifCondPrior


"""
test mk experiment API with "diva" model and "mldg" trainer
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
zd_dim = 3
zy_dim = 10
zx_dim = 0 # TODO: specify zx_dim
chain_node_builder = ChainNodeVAEBuilderClassifCondPrior(successor_node=None).init_business(zd_dim, zx_dim, zy_dim) # TODO: fix chain_node_builder?
list_str_y = [f"class{i}" for i in range(zy_dim)]
list_d_tr = ["domain2", "domain3"]
gamma_d = 0
gamma_y = 0
beta_d = 0
beta_x = 0
beta_y = 0

# specify model to use
model = mk_diva()(chain_node_builder, zd_dim, zy_dim, zx_dim, list_str_y, list_d_tr, gamma_d, gamma_y, beta_d, beta_x, beta_y)

# make trainer for model
exp = mk_exp(task, model, trainer="mldg", test_domain="domain1", batchsize=32)
exp.execute(num_epochs=3)
