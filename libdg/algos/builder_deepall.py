import copy
from libdg.algos.a_algo_builder import NodeAlgoBuilder
from libdg.algos.trainers.train_basic import TrainerBasic
from libdg.algos.msels.c_msel import MSelTrLoss
from libdg.algos.msels.c_msel_oracle import MSelOracleVisitor
from libdg.algos.observers.b_obvisitor import ObVisitor
from libdg.utils.utils_cuda import get_device
from libdg.compos.nn_alex import Alex4DeepAll
from libdg.models.model_deep_all import ModelDeepAll
from libdg.compos.zoo_nn import FeatExtractNNBuilderChainNodeGetter


class NodeAlgoBuilderDeepAll(NodeAlgoBuilder):
    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        device = get_device(args.nocu)
        observer = ObVisitor(exp,
                             MSelOracleVisitor(
                                 MSelTrLoss(max_es=args.es)), device)

        net = FeatExtractNNBuilderChainNodeGetter(args)()  # request
        net.init_business()

        model = ModelDeepAll(net, list_str_y=task.list_str_y)
        trainer = TrainerBasic(model, task, observer, device, args)
        return trainer
