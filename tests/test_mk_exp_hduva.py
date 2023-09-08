"""
make an experiment
"""
from domainlab.mk_exp import mk_exp
from domainlab.dsets.dset_mnist_color_solo_default import DsetMNISTColorSoloDefault
from domainlab.tasks.task_dset import mk_task_dset
from domainlab.models.model_hduva import mk_hduva
from domainlab.tasks.utils_task import ImSize
from domainlab.compos.pcr.request import RequestVAEBuilderNN
from domainlab.compos.vae.compos.encoder import LSEncoderConvBnReluPool
from domainlab.compos.vae.utils_request_chain_builder import VAEChainNodeGetter


def test_mk_exp_hduva():
    """
    test mk experiment API with "hduva" model and trainers "mldg", "diva"
    """

    mk_exp_hduva(trainer="mldg")
    mk_exp_hduva(trainer="diva")


def mk_exp_hduva(trainer="mldg"):
    """
    execute experiment with "hduva" model and custom trainer
    """

    # specify domain generalization task
    task = mk_task_dset(isize=ImSize(3, 28, 28), dim_y=10, taskna="custom_task")
    task.add_domain(name="domain1",
                    dset_tr=DsetMNISTColorSoloDefault(0),
                    dset_val=DsetMNISTColorSoloDefault(1))
    task.add_domain(name="domain2",
                    dset_tr=DsetMNISTColorSoloDefault(2),
                    dset_val=DsetMNISTColorSoloDefault(3))
    task.add_domain(name="domain3",
                    dset_tr=DsetMNISTColorSoloDefault(4),
                    dset_val=DsetMNISTColorSoloDefault(5))

    # specify backbone to use
    list_str_y = [f"class{i}" for i in range(task.dim_y)]
    list_d_tr = ["domain2", "domain3"]
    gamma_d = 1e5
    gamma_y = 7e5
    beta_d = 1e3
    beta_x = 1e3
    beta_y = 1e3
    beta_t = 1e3
    device = "cpu"
    zx_dim = 30
    zy_dim = 10
    zd_dim = 3
    topic_dim = 3

    net_class_d = LSEncoderConvBnReluPool(
        zd_dim, task.isize.c, task.isize.w, task.isize.h, conv_stride=1)
    net_x = LSEncoderConvBnReluPool(
        zx_dim, task.isize.c, task.isize.w, task.isize.h, conv_stride=1)
    net_class_y = LSEncoderConvBnReluPool(
        zy_dim, task.isize.c, task.isize.w, task.isize.h, conv_stride=1)

    request = RequestVAEBuilderNN(net_class_d, net_x, net_class_y,
                                  task.isize.c, task.isize.h, task.isize.w)
    chain_node_builder = VAEChainNodeGetter(request)()

    # specify model to use
    model = mk_hduva()(chain_node_builder, zy_dim, zd_dim, list_str_y, list_d_tr, gamma_d, gamma_y,
                       beta_d, beta_x, beta_y, beta_t, device, zx_dim, topic_dim)

    # make trainer for model
    exp = mk_exp(task, model, trainer=trainer, test_domain="domain1", batchsize=32)
    exp.execute(num_epochs=2)