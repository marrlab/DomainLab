import copy
from libdg.algos.a_algo_builder import NodeAlgoBuilder
from libdg.algos.trainers.train_basic import TrainerBasic
from libdg.algos.msels.c_msel import MSelTrLoss
from libdg.algos.msels.c_msel_oracle import MSelOracleVisitor
from libdg.algos.observers.b_obvisitor import ObVisitor
from libdg.utils.utils_cuda import get_device
from libdg.compos.nn_alex import Alex4DeepAll
from libdg.models.model_deep_all import ModelDeepAll
from libdg.utils.u_import_net_module import \
    build_external_obj_net_module_feat_extract, import_net_module_from_path


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
        if args.npath is not None:  # custom network user specified from file
            module = import_net_module_from_path(args.npath)
            # import NetFeatExtract as module
            # net = module(flag_pretrain=True, dim_y=task.dim_y)
            net = build_external_obj_net_module_feat_extract(
                args.npath, task.dim_y, task.isize.i_c, task.isize.i_h, task.isize.i_w)
            net = copy.deepcopy(net)
        else:
            net = Alex4DeepAll(flag_pretrain=True, dim_y=task.dim_y)
        model = ModelDeepAll(net, list_str_y=task.list_str_y)
        trainer = TrainerBasic(model, task, observer, device, args)
        return trainer
