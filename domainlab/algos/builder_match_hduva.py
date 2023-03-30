"""
hduva with matchdg
"""
from domainlab.algos.a_algo_builder import NodeAlgoBuilder
from domainlab.algos.msels.c_msel import MSelTrLoss
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.b_obvisitor import ObVisitor
from domainlab.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from domainlab.algos.trainers.train_visitor import TrainerVisitor
from domainlab.compos.pcr.request import RequestVAEBuilderCHW
from domainlab.compos.vae.utils_request_chain_builder import VAEChainNodeGetter
from domainlab.models.model_hduva import mk_hduva
from domainlab.utils.utils_cuda import get_device

from domainlab.algos.trainers.train_matchdg import TrainerMatchDG
from domainlab.models.wrapper_matchdg import ModelWrapMatchDGLogit
from domainlab.models.model_wrapper_matchdg4net import ModelWrapMatchDGNet


class NodeAlgoBuilderMatchHDUVA(NodeAlgoBuilder):
    """
    NodeAlgoBuilderMatchHDUVA
    """
    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        request = RequestVAEBuilderCHW(
            task.isize.c, task.isize.h, task.isize.w, args)
        device = get_device(args.nocu)
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

        model = ModelWrapMatchDGLogit(model, list_str_y=task.list_str_y)
        model = model.to(device)

        ctr_model = ModelWrapMatchDGNet(model, list_str_y=task.list_str_y)
        ctr_model = ctr_model.to(device)

        model_sel = MSelOracleVisitor(MSelTrLoss(max_es=args.es))
        observer = ObVisitor(exp,
                             model_sel,
                             device)

        trainer = TrainerMatchDG(exp, task, ctr_model, model, observer, args,
                                 device)

        return trainer
