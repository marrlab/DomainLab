"""
builder for JiGen
"""
from domainlab.algos.a_algo_builder import NodeAlgoBuilder
from domainlab.algos.msels.c_msel_val import MSelValPerf
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.b_obvisitor import ObVisitor
from domainlab.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from domainlab.algos.trainers.train_hyper_scheduler import TrainerHyperScheduler
from domainlab.algos.trainers.hyper_scheduler import HyperSchedulerWarmupExponential
from domainlab.algos.trainers.zoo_trainer import TrainerChainNodeGetter
from domainlab.compos.nn_zoo.net_classif import ClassifDropoutReluLinear
from domainlab.compos.utils_conv_get_flat_dim import get_flat_dim
from domainlab.compos.zoo_nn import FeatExtractNNBuilderChainNodeGetter
from domainlab.models.model_jigen import mk_jigen
from domainlab.utils.utils_cuda import get_device
from domainlab.dsets.utils_wrapdset_patches import WrapDsetPatches


class NodeAlgoBuilderJiGen(NodeAlgoBuilder):
    """
    NodeAlgoBuilderJiGen
    """
    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        device = get_device(args)
        msel = MSelOracleVisitor(msel=MSelValPerf(max_es=args.es))
        observer = ObVisitor(msel)
        observer = ObVisitorCleanUp(observer)

        builder = FeatExtractNNBuilderChainNodeGetter(
            args, arg_name_of_net="nname",
            arg_path_of_net="npath")()  # request, @FIXME, constant string

        net_encoder = builder.init_business(
            flag_pretrain=True, dim_out=task.dim_y,
            remove_last_layer=False, args=args,
            i_c=task.isize.i_c,
            i_w=task.isize.i_w,
            i_h=task.isize.i_h)

        dim_feat = get_flat_dim(net_encoder,
                                task.isize.i_c,
                                task.isize.i_h,
                                task.isize.i_w)

        net_classifier = ClassifDropoutReluLinear(dim_feat, task.dim_y)

        # @FIXME: this seems to be the only difference w.r.t. builder_dann
        net_classifier_perm = ClassifDropoutReluLinear(
            dim_feat, args.nperm+1)
        model = mk_jigen()(list_str_y=task.list_str_y,
                           coeff_reg=args.gamma_reg,
                           net_encoder=net_encoder,
                           net_classifier_class=net_classifier,
                           net_classifier_permutation=net_classifier_perm)

        trainer = TrainerChainNodeGetter(args.trainer)(default="hyperscheduler")
        trainer.init_business(model, task, observer, device, args)
        if isinstance(trainer, TrainerHyperScheduler):
            trainer.set_scheduler(HyperSchedulerWarmupExponential,
                                  total_steps=trainer.num_batches*args.epos,
                                  flag_update_epoch=False,
                                  flag_update_batch=True)
        return trainer, model, observer, device
