import copy
from libdg.algos.a_algo_builder import NodeAlgoBuilder
from libdg.algos.observers.b_obvisitor import ObVisitor
from libdg.algos.msels.c_msel_oracle import MSelOracleVisitor
from libdg.algos.msels.c_msel import MSelTrLoss
from libdg.algos.trainers.train_matchdg import TrainerMatchDG
from libdg.compos.nn_alex import AlexNetNoLastLayer, Alex4DeepAll
from libdg.models.model_deep_all import ModelDeepAll
from libdg.models.wrapper_matchdg import ModelWrapMatchDGLogit
from libdg.utils.utils_cuda import get_device
from libdg.utils.u_import_net_module import \
    build_external_obj_net_module_feat_extract, import_net_module_from_path


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
            get_ctr_model_erm_creator(args,
                task.isize.c, task.isize.h, task.isize.w, task.dim_y, task.list_str_y)
        model = fun_build_erm_phi()
        model = model.to(device)
        ctr_model = fun_build_ctr()
        ctr_model = ctr_model.to(device)
        observer = ObVisitor(exp, MSelOracleVisitor(MSelTrLoss(max_es=args.es)), device)
        trainer = TrainerMatchDG(exp, task, ctr_model, model, observer, args, device)
        return trainer


def get_ctr_model_erm_creator(args, i_c, i_h, i_w, dim_y, list_str_y):
    def fun_build_alex_erm():
        if args.npath is not None:  # custom network user specified from file
            module = import_net_module_from_path(args.npath)
            # import NetFeatExtract as module
            # net = module(flag_pretrain=True, dim_y=task.dim_y)
            net = build_external_obj_net_module_feat_extract(
                args.npath, task.dim_y, task.isize.i_c, task.isize.i_h, task.isize.i_w)
            net = copy.deepcopy(net)
        else:
            net = Alex4DeepAll(flag_pretrain=True, dim_y=task.dim_y)
        model = ModelDeepAll(net, list_str_y=list_str_y)
        model = ModelWrapMatchDGLogit(model, list_str_y=list_str_y)
        return model

    def fun_build_alex_ctr():
        return AlexNetNoLastLayer(flag_pretrain=True)

    if i_h > 100:
        return fun_build_alex_ctr, fun_build_alex_erm
    raise NotImplementedError
