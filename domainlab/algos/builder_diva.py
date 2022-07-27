from domainlab.algos.a_algo_builder import NodeAlgoBuilder
from domainlab.algos.trainers.train_visitor import TrainerVisitor
from domainlab.algos.msels.c_msel import MSelTrLoss
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.b_obvisitor import ObVisitor
from domainlab.algos.observers.c_obvisitor_gen import ObVisitorGen
from domainlab.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from domainlab.compos.vae.utils_request_chain_builder import VAEChainNodeGetter
from domainlab.compos.pcr.request import RequestVAEBuilderCHW
from domainlab.models.model_diva import ModelDIVA
from domainlab.utils.utils_cuda import get_device


class NodeAlgoBuilderDIVA(NodeAlgoBuilder):
    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        request = RequestVAEBuilderCHW(
            task.isize.c, task.isize.h, task.isize.w, args)
        node = VAEChainNodeGetter(request)()
        model = ModelDIVA(node,
                          zd_dim=args.zd_dim, zy_dim=args.zy_dim,
                          zx_dim=args.zx_dim,
                          list_str_y=task.list_str_y,
                          list_d_tr=task.list_domain_tr,
                          gamma_d=args.gamma_d,
                          gamma_y=args.gamma_y,
                          beta_x=args.beta_x,
                          beta_y=args.beta_y,
                          beta_d=args.beta_d)
        device = get_device(args.nocu)
        if not args.gen:
            observer = ObVisitorCleanUp(
                ObVisitor(exp,
                          MSelOracleVisitor(MSelTrLoss(max_es=args.es)),
                          device))
        else:
            observer = ObVisitorCleanUp(
                ObVisitorGen(exp,
                             MSelOracleVisitor(MSelTrLoss(max_es=args.es)),
                             device))
        trainer = TrainerVisitor(model, task, observer, device, args)
        return trainer
