from libdg.algos.a_algo_builder import NodeAlgoBuilder
from libdg.algos.trainers.train_diva import TrainerDIVA
from libdg.algos.msels.c_msel import MSelTrLoss
from libdg.algos.msels.c_msel_oracle import MSelOracleVisitor
from libdg.algos.observers.b_obvisitor import ObVisitor
from libdg.algos.observers.c_obvisitor_gen import ObVisitorGen
from libdg.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from libdg.compos.vae.utils_request_chain_builder import VAEChainNodeGetter
from libdg.compos.pcr.request import RequestVAEBuilderCHW
from libdg.models.model_diva import ModelDIVA
from libdg.utils.utils_cuda import get_device


class NodeAlgoBuilderDIVA(NodeAlgoBuilder):
    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        request = RequestVAEBuilderCHW(task.isize.c, task.isize.h, task.isize.w)
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
                ObVisitor(exp, MSelOracleVisitor(MSelTrLoss(max_es=args.es)), device))
        else:
            observer = ObVisitorCleanUp(
                ObVisitorGen(exp, MSelOracleVisitor(MSelTrLoss(max_es=args.es)), device))
        trainer = TrainerDIVA(model, task, observer, device, aconf=args)
        return trainer
