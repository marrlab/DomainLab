from domainlab.algos.a_algo_builder import NodeAlgoBuilder
from domainlab.algos.msels.c_msel import MSelTrLoss
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.b_obvisitor import ObVisitor
from domainlab.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from domainlab.algos.trainers.train_matchdg import TrainerMatchDG
from domainlab.compos.zoo_nn import FeatExtractNNBuilderChainNodeGetter
from domainlab.models.model_deep_all import mk_deepall
from domainlab.models.wrapper_matchdg import ModelWrapMatchDGLogit
from domainlab.utils.utils_cuda import get_device


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

        erm_builder = FeatExtractNNBuilderChainNodeGetter(
            args,
            arg_name_of_net="nname",
            arg_path_of_net="npath")()  # request, #FIXME: constant string

        erm_net = erm_builder.init_business(
            flag_pretrain=True, dim_out=task.dim_y,
            remove_last_layer=False, args=args,
            i_c=task.isize.i_c, i_h=task.isize.i_h,
            i_w=task.isize.i_w)

        model = mk_deepall(erm_net, list_str_y=task.list_str_y)
        model = ModelWrapMatchDGLogit(model, list_str_y=task.list_str_y)

        model = model.to(device)
        ctr_builder = FeatExtractNNBuilderChainNodeGetter(
            args,
            arg_name_of_net="nname",
            arg_path_of_net="npath")()  # request, #FIXME constant string

        ctr_net = ctr_builder.init_business(
            flag_pretrain=True,
            dim_out=task.dim_y,
            # FIXME: ctr_model should not rely on task.dim_y so it could support more tasks? maybe use task.feat_num?
            remove_last_layer=True,
            args=args,
            i_c=task.isize.i_c,
            i_h=task.isize.i_h,
            i_w=task.isize.i_w)
        ctr_model = ctr_net.to(device)

        model_sel = MSelOracleVisitor(MSelTrLoss(max_es=args.es))
        observer = ObVisitor(exp,
                             model_sel,
                             device)
        trainer = TrainerMatchDG(exp, task, ctr_model, model, observer, args,
                                 device)
        return trainer
