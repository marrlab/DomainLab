from domainlab.algos.a_algo_builder import NodeAlgoBuilder
from domainlab.algos.msels.c_msel import MSelTrLoss
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.b_obvisitor import ObVisitor
from domainlab.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from domainlab.algos.trainers.train_visitor import TrainerVisitor
from domainlab.algos.trainers.zoo_trainer import TrainerChainNodeGetter
from domainlab.compos.pcr.request import RequestVAEBuilderCHW
from domainlab.compos.vae.utils_request_chain_builder import VAEChainNodeGetter
from domainlab.models.model_hduva import mk_hduva
from domainlab.utils.utils_cuda import get_device


class NodeAlgoBuilderHDUVA(NodeAlgoBuilder):
    """
    NodeAlgoBuilderHDUVA
    """
    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        request = RequestVAEBuilderCHW(
            task.isize.c, task.isize.h, task.isize.w, args)
        device = get_device(args)
        node = VAEChainNodeGetter(request, args.topic_dim)()
        model = mk_hduva()(node,
                           zd_dim=args.zd_dim,
                           zy_dim=args.zy_dim,
                           zx_dim=args.zx_dim,
                           device=device,
                           topic_dim=args.topic_dim,
                           list_str_y=task.list_str_y,
                           list_d_tr=task.list_domain_tr,
                           gamma_d=args.gamma_d,
                           gamma_y=args.gamma_y,
                           beta_t=args.beta_t,
                           beta_x=args.beta_x,
                           beta_y=args.beta_y,
                           beta_d=args.beta_d)
        model_sel = MSelOracleVisitor(MSelTrLoss(max_es=args.es))
        observer = ObVisitorCleanUp(
            ObVisitor(exp, model_sel, device))
        trainer = TrainerChainNodeGetter(args)(default="visitor")
        trainer.init_business(model, task, observer, device, args)
        return trainer
