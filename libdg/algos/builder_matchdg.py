from libdg.algos.a_algo_builder import NodeAlgoBuilder
from libdg.algos.observers.b_obvisitor import ObVisitor
from libdg.algos.msels.c_msel_oracle import MSelOracleVisitor
from libdg.algos.msels.c_msel import MSelTrLoss
from libdg.algos.trainers.train_matchdg import TrainerMatchDG
from libdg.compos.nn_alex import AlexNetNoLastLayer, Alex4DeepAll
from libdg.models.model_deep_all import ModelDeepAll
from libdg.models.wrapper_matchdg import ModelWrapMatchDGLogit
from libdg.utils.utils_cuda import get_device


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
        fun_build_ctr, fun_build_erm_phi = \
            get_ctr_model_erm_creator(
                task.isize.c, task.isize.h, task.isize.w, task.dim_y, task.list_str_y)
        model = fun_build_erm_phi()
        model = model.to(device)
        ctr_model = fun_build_ctr()
        ctr_model = ctr_model.to(device)
        observer = ObVisitor(exp, MSelOracleVisitor(MSelTrLoss(max_es=args.es)), device)
        trainer = TrainerMatchDG(exp, task, ctr_model, model, observer, args, device)
        return trainer


def get_ctr_model_erm_creator(i_c, i_h, i_w, dim_y, list_str_y):
    def fun_build_alex_erm():
        net = Alex4DeepAll(flag_pretrain=True, dim_y=dim_y)
        model = ModelDeepAll(net, list_str_y=list_str_y)
        model = ModelWrapMatchDGLogit(model, list_str_y=list_str_y)
        return model

    def fun_build_alex_ctr():
        return AlexNetNoLastLayer(flag_pretrain=True)

    if i_h > 100:
        return fun_build_alex_ctr, fun_build_alex_erm
    raise NotImplementedError
