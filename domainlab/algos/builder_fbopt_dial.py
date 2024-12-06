"""
builder for feedback optimization of dial
"""
from domainlab.algos.builder_diva import NodeAlgoBuilderDIVA
from domainlab.algos.trainers.train_fbopt_b import TrainerFbOpt


class NodeAlgoBuilderFbOptDial(NodeAlgoBuilderDIVA):
    """
    builder for feedback optimization for dial
    """

    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        trainer_in, model, observer, device = super().init_business(exp)
        trainer_in.init_business(model, exp.task, observer, device, exp.args)
        trainer = TrainerFbOpt()
        trainer.init_business(trainer_in, exp.task, observer, device, exp.args)
        return trainer, model, observer, device
