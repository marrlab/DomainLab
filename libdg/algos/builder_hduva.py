from libdg.algos.a_algo_builder import NodeAlgoBuilder
from libdg.algos.trainers.train_basic import TrainerBasic
from libdg.algos.msels.c_msel import MSelTrLoss
from libdg.algos.msels.c_msel_oracle import MSelOracleVisitor
from libdg.algos.observers.b_obvisitor import ObVisitor
from libdg.algos.observers.c_obvisitor_gen import ObVisitorGen
from libdg.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from libdg.compos.vae.utils_request_chain_builder import VAEChainNodeGetter
from libdg.compos.pcr.request import RequestVAEBuilderCHW
from libdg.models.model_hduva import ModelHDUVA
from libdg.utils.utils_cuda import get_device


class NodeAlgoBuilderHDUVA(NodeAlgoBuilder):
    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        request = RequestVAEBuilderCHW(task.isize.c, task.isize.h, task.isize.w)
        device = get_device(args.nocu)
        node = VAEChainNodeGetter(request, args.topic_dim)()
        model = ModelHDUVA(node,
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
        observer = ObVisitorCleanUp(
            ObVisitor(exp, MSelOracleVisitor(MSelTrLoss(max_es=args.es)), device))
        trainer = TrainerBasic(model, task, observer, device, args)
        return trainer
