"""
builder for erm
"""
from domainlab.algos.utils import split_net_feat_last
from domainlab.algos.a_algo_builder import NodeAlgoBuilder
from domainlab.algos.msels.c_msel_val import MSelValPerf
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.b_obvisitor import ObVisitor
from domainlab.algos.trainers.zoo_trainer import TrainerChainNodeGetter
from domainlab.compos.zoo_nn import FeatExtractNNBuilderChainNodeGetter
from domainlab.models.model_erm import mk_erm
from domainlab.utils.utils_cuda import get_device


class NodeAlgoBuilderERM(NodeAlgoBuilder):
    """
    builder for erm
    """
    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        device = get_device(args)
        model_sel = MSelOracleVisitor(MSelValPerf(max_es=args.es))
        observer = ObVisitor(model_sel)

        builder = FeatExtractNNBuilderChainNodeGetter(
            args, arg_name_of_net="nname",
            arg_path_of_net="npath")()  # request, # @FIXME, constant string

        net = builder.init_business(flag_pretrain=True, dim_out=task.dim_y,
                                    remove_last_layer=False, args=args,
                                    isize=(task.isize.i_c, task.isize.i_h, task.isize.i_w))

        _, _ = split_net_feat_last(net)

        model = mk_erm()(
                net=net,
                # net_feat=net_invar_feat, net_classifier=net_classifier,
                list_str_y=task.list_str_y)
        model = self.init_next_model(model, exp)
        trainer = TrainerChainNodeGetter(args.trainer)(default="basic")
        # trainer.init_business(model, task, observer, device, args)
        return trainer, model, observer, device
