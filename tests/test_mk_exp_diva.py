"""
make an experiment using "diva" model
"""

from domainlab.mk_exp import mk_exp
from domainlab.dsets.dset_mnist_color_solo_default import DsetMNISTColorSoloDefault
from domainlab.tasks.task_dset import mk_task_dset
from domainlab.models.model_diva import mk_diva
from domainlab.tasks.utils_task import ImSize
from domainlab.compos.vae.utils_request_chain_builder import VAEChainNodeGetter
from domainlab.compos.pcr.request import RequestVAEBuilderCHW


def test_mk_exp_diva():
    """
    test mk experiment API for "diva" model and trainers "mldg", "dial"
    """

    mk_exp_diva(trainer="mldg")
    mk_exp_diva(trainer="dial")


def mk_exp_diva(trainer="mldg"):
    """
    execute experiment with "diva" model and custom trainer
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
    zx_dim = 30
    args = f"--aname=apimodel --trainer={trainer} --te_d=domain1 --bs=32"
    request = RequestVAEBuilderCHW(task.isize.c, task.isize.h, task.isize.w, args)
    chain_node_builder = VAEChainNodeGetter(request)()  # TODO: chain_node_builder
    list_str_y = [f"class{i}" for i in range(task.dim_y)]
    list_d_tr = ["domain2", "domain3"]
    gamma_d = 1e5
    gamma_y = 7e5
    beta_d = 1e3
    beta_x = 1e3
    beta_y = 1e3

    # specify model to use
    model = mk_diva()(chain_node_builder, zd_dim, zy_dim, zx_dim, list_str_y, list_d_tr, gamma_d,
                      gamma_y, beta_d, beta_x, beta_y)

    # make trainer for model
    exp = mk_exp(task, model, trainer=trainer, test_domain="domain1", batchsize=32)
    exp.execute(num_epochs=2)
