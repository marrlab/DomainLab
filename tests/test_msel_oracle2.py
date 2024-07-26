"""
executing mk_exp multiple times will cause deep copy to be called multiple times, pytest will show process got killed.
"""
from tests.utils_task_model import mk_exp, mk_model, mk_task


def test_msel_oracle4():
    """
    return trainer, model, observer
    """
    task = mk_task()
    model = mk_model(task)
    # specify backbone to use
    exp = mk_exp(
        task,
        model,
        trainer="mldg",
        test_domain="domain1",
        batchsize=2,
        alone=False,
        msel_loss_tr=True,
    )
    exp.execute(num_epochs=2)
    exp.trainer.observer.model_sel.msel.best_loss = 0
    exp.trainer.observer.model_sel.msel.update(epoch=1, clear_counter=True)
    del exp
