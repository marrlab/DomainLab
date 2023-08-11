"""
example task construction:
    Specify each domain by a training set and validation (can be None)
"""
from domainlab.tasks.task_dset import mk_task_dset
from domainlab.tasks.utils_task import ImSize
from domainlab.dsets.dset_mnist_color_solo_default import DsetMNISTColorSoloDefault


task = mk_task_dset(isize=ImSize(3, 28, 28),  taskna="custom_task")
task.add_domain(name="domain1",
                dset_tr=DsetMNISTColorSoloDefault(0),
                dset_val=DsetMNISTColorSoloDefault(1))
task.add_domain(name="domain2",
                dset_tr=DsetMNISTColorSoloDefault(2),
                dset_val=DsetMNISTColorSoloDefault(3))
task.add_domain(name="domain3",
                dset_tr=DsetMNISTColorSoloDefault(4),
                dset_val=DsetMNISTColorSoloDefault(5))


def get_task(na=None):
    return task
