import copy
from libdg.algos.a_algo_builder import NodeAlgoBuilder
from libdg.algos.observers.b_obvisitor import ObVisitor
from libdg.algos.msels.c_msel_oracle import MSelOracleVisitor
from libdg.algos.msels.c_msel import MSelTrLoss
from libdg.algos.trainers.train_matchdg import TrainerMatchDG
from libdg.utils.utils_cuda import get_device
from libdg.compos.zoo_nn import FeatExtractNNBuilderChainNodeGetter


class NodeAlgoBuilderMatchDG(NodeAlgoBuilder):
    """
    """
    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        device = get_device(args.nocu)

        erm_builder = FeatExtractNNBuilderChainNodeGetter(args)()  # request
        erm_net = erm_builder.init_business(
            flag_pretrain=True, dim_feat=task.dim_y,
            remove_last_layer=False, args=args)
        model = erm_net.to(device)

        ctr_builder = FeatExtractNNBuilderChainNodeGetter(args)()  # request
        ctr_net = ctr_builder.init_business(
            flag_pretrain=True, dim_feat=task.dim_y,
            remove_last_layer=True, args=args)
        ctr_model = ctr_net.to(device)

        observer = ObVisitor(exp,
                             MSelOracleVisitor(MSelTrLoss(max_es=args.es)),
                             device)
        trainer = TrainerMatchDG(exp, task, ctr_model, model, observer, args,
                                 device)
        return trainer
