"""
build algorithm from API coded model with custom backbone
"""
from domainlab.algos.a_algo_builder import NodeAlgoBuilder
from domainlab.algos.trainers.zoo_trainer import TrainerChainNodeGetter
from domainlab.algos.msels.c_msel_val import MSelValPerf
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.b_obvisitor import ObVisitor
from domainlab.utils.utils_cuda import get_device


class NodeAlgoBuilderAPIModel(NodeAlgoBuilder):
    """
    build algorithm from API coded model with custom backbone
    """
    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        args = exp.args
        device = get_device(args)
        model_sel = MSelOracleVisitor(MSelValPerf(max_es=args.es))
        observer = ObVisitor(model_sel) 
        trainer = TrainerChainNodeGetter(args.trainer)(default="hyperscheduler")
        return trainer, None, observer, device
