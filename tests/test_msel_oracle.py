"""
executing mk_exp multiple times will cause deep copy to be called multiple times, pytest will show process got killed.
"""
from tests.utils_task_model import mk_exp, mk_model, mk_task

def test_msel_oracle():
    """
    return trainer, model, observer
    """
    # specify backbone to use
    task = mk_task()
    model = mk_model(task)
    # make trainer for model
    exp = mk_exp(task, model, trainer="mldg", test_domain="domain1", batchsize=2)
    exp.execute(num_epochs=2)

    del exp


def test_msel_oracle1():
    """
    return trainer, model, observer
    """
    task = mk_task()
    model = mk_model(task)
    exp = mk_exp(
        task, model, trainer="mldg", test_domain="domain1", batchsize=2, alone=False
    )

    exp.execute(num_epochs=2)
    exp.trainer.observer.model_sel.msel.update(epoch=1, clear_counter=True)
    del exp


def test_msel_oracle2():
    """
    return trainer, model, observer
    """
    task = mk_task()
    model = mk_model(task)

    # make trainer for model
    exp = mk_exp(task, model, trainer="mldg", test_domain="domain1", batchsize=2)
    exp.execute(num_epochs=2)


def test_msel_oracle3():
    """
    return trainer, model, observer
    """
    task = mk_task()
    model = mk_model(task)

    exp = mk_exp(
        task,
        model,
        trainer="mldg",
        test_domain="domain1",
        batchsize=2,
        alone=False,
        force_best_val=True,
    )
    exp.execute(num_epochs=2)
    del exp
