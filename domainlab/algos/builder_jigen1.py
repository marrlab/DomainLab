"""
builder for JiGen
"""
from domainlab.algos.a_algo_builder import NodeAlgoBuilder
from domainlab.algos.msels.c_msel import MSelTrLoss
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.b_obvisitor import ObVisitor
from domainlab.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from domainlab.algos.trainers.train_visitor import TrainerVisitor
from domainlab.algos.trainers.train_visitor import HyperSchedulerAneal
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
    def dset_decoration_args_algo(self, args, ddset):
        """
        JiGen need to shuffle the tiles of the original image
        """
        ddset = WrapDsetPatches(ddset,
                                num_perms2classify=args.nperm,
                                prob_no_perm=1-args.pperm,
                                grid_len=args.grid_len,
                                ppath=args.jigen_ppath)
        return ddset

    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        device = get_device(args)
        msel = MSelOracleVisitor(MSelTrLoss(max_es=args.es))
        observer = ObVisitor(exp, msel, device)
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
                           list_str_d=task.list_domain_tr,
                           coeff_reg=args.gamma_reg,
                           net_encoder=net_encoder,
                           net_classifier_class=net_classifier,
                           net_classifier_permutation=net_classifier_perm)

        trainer = TrainerVisitor()
        trainer.init_business(model, task, observer, device, args)
        trainer.set_scheduler(HyperSchedulerAneal,
                              total_steps=trainer.num_batches*args.epos,
                              flag_update_epoch=False,
                              flag_update_batch=True)
        return trainer
