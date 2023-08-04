"""
build algorithm from API coded model with custom backbone
"""
from domainlab.algos.a_algo_builder import NodeAlgoBuilder
from domainlab.algos.trainers.zoo_trainer import TrainerChainNodeGetter


class NodeAlgoBuilderAPIModel(NodeAlgoBuilder):
    """
    build algorithm from API coded model with custom backbone
    """
    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        trainer = TrainerChainNodeGetter(args)(default="visitor")
        return trainer
