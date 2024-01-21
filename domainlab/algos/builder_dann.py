"""
builder for Domain Adversarial Neural Network: accept different training scheme
"""
from domainlab.algos.a_algo_builder import NodeAlgoBuilder
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.msels.c_msel_val import MSelValPerf
from domainlab.algos.observers.b_obvisitor import ObVisitor
from domainlab.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from domainlab.algos.trainers.hyper_scheduler import HyperSchedulerWarmupExponential
from domainlab.algos.trainers.zoo_trainer import TrainerChainNodeGetter
from domainlab.compos.nn_zoo.net_classif import ClassifDropoutReluLinear
from domainlab.compos.utils_conv_get_flat_dim import get_flat_dim
from domainlab.compos.zoo_nn import FeatExtractNNBuilderChainNodeGetter
from domainlab.models.model_dann import mk_dann
from domainlab.utils.utils_cuda import get_device


class NodeAlgoBuilderDANN(NodeAlgoBuilder):
    """
    NodeAlgoBuilderDANN
    """

    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        self._task = task
        args = exp.args
        task.get_list_domains_tr_te(args.tr_d, args.te_d)
        device = get_device(args)
        msel = MSelOracleVisitor(MSelValPerf(max_es=args.es))
        observer = ObVisitor(msel)
        observer = ObVisitorCleanUp(observer)

        builder = FeatExtractNNBuilderChainNodeGetter(
            args, arg_name_of_net="nname", arg_path_of_net="npath"
        )()  # request, @FIXME, constant string

        net_encoder = builder.init_business(
            flag_pretrain=True,
            dim_out=task.dim_y,
            remove_last_layer=False,
            args=args,
            isize=(task.isize.i_c, task.isize.i_w, task.isize.i_h),
        )

        dim_feat = get_flat_dim(
            net_encoder, task.isize.i_c, task.isize.i_h, task.isize.i_w
        )

        net_classifier = ClassifDropoutReluLinear(dim_feat, task.dim_y)

        net_discriminator = self.reset_aux_net(net_encoder)
        model = mk_dann(list_str_y=task.list_str_y,
                        net_classifier=net_classifier)(
            list_d_tr=task.list_domain_tr,
            alpha=args.gamma_reg,
            net_encoder=net_encoder,
            net_discriminator=net_discriminator,
            builder=self)

        model = self.init_next_model(model, exp)
        trainer = TrainerChainNodeGetter(args.trainer)(default="hyperscheduler")
        trainer.init_business(model, task, observer, device, args)
        if trainer.name == "hyperscheduler":
            trainer.set_scheduler(
                HyperSchedulerWarmupExponential,
                total_steps=trainer.num_batches * args.warmup,
                flag_update_epoch=False,
                flag_update_batch=True,
            )
        return trainer, model, observer, device

    def reset_aux_net(self, net_encoder):
        """
        reset auxilliary neural network from task
        note that net_encoder can also be a method like extract_semantic_feat
        """
        dim_feat = get_flat_dim(
            net_encoder,
            self._task.isize.i_c,
            self._task.isize.i_h,
            self._task.isize.i_w,
        )
        net_discriminator = ClassifDropoutReluLinear(
            dim_feat, len(self._task.list_domain_tr)
        )
        return net_discriminator
