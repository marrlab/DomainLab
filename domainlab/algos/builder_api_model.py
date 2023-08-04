"""
build algorithm from API coded model with custom backbone
"""
from domainlab.algos.a_algo_builder import NodeAlgoBuilder
from domainlab.algos.msels.c_msel_val import MSelValPerf
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.b_obvisitor import ObVisitor
from domainlab.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from domainlab.algos.trainers.zoo_trainer import TrainerChainNodeGetter
from domainlab.compos.pcr.request import RequestVAEBuilderCHW
from domainlab.compos.vae.utils_request_chain_builder import VAEChainNodeGetter
from domainlab.models.model_hduva import mk_hduva
from domainlab.utils.utils_cuda import get_device


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
        device = get_device(args)
        model_sel = MSelValPerf(max_es=args.es)
        observer = ObVisitorCleanUp(
            ObVisitor(exp, model_sel, device))
        trainer = TrainerChainNodeGetter(args)(default="visitor")
        return trainer
