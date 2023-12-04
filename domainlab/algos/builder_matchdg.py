"""
builder for mathcdg with deepall
"""
import copy
from domainlab.algos.a_algo_builder import NodeAlgoBuilder
from domainlab.algos.msels.c_msel_val import MSelValPerf
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.b_obvisitor import ObVisitor
from domainlab.algos.trainers.train_matchdg import TrainerMatchDG
from domainlab.compos.zoo_nn import FeatExtractNNBuilderChainNodeGetter
from domainlab.models.model_deep_all import mk_deepall
from domainlab.utils.utils_cuda import get_device
from domainlab.tasks.utils_task_dset import DsetIndDecorator4XYD


class NodeAlgoBuilderMatchDG(NodeAlgoBuilder):
    """
    algorithm builder for matchDG
    """
    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        device = get_device(args)

        erm_builder = FeatExtractNNBuilderChainNodeGetter(
            args,
            arg_name_of_net="nname",
            arg_path_of_net="npath")()  # request, # @FIXME: constant string

        erm_net = erm_builder.init_business(
            flag_pretrain=True,
            dim_out=task.dim_y,
            remove_last_layer=False,
            args=args,
            i_c=task.isize.i_c,
            i_h=task.isize.i_h,
            i_w=task.isize.i_w)

        model = mk_deepall()(erm_net, list_str_y=task.list_str_y)
        # model = ModelWrapMatchDGLogit(model, list_str_y=task.list_str_y)
        model = model.to(device)

        # different than model, ctr_model has no classification loss

        model_sel = MSelOracleVisitor(MSelValPerf(max_es=args.es))
        observer = ObVisitor(model_sel)

        trainer = TrainerMatchDG()
        trainer.init_business(model, task, observer, device, args)
        return trainer, model, observer, device
