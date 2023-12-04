"""
hduva with matchdg
"""
import copy

from domainlab.algos.a_algo_builder import NodeAlgoBuilder
from domainlab.algos.msels.c_msel_val import MSelValPerf
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.b_obvisitor import ObVisitor
from domainlab.tasks.utils_task_dset import DsetIndDecorator4XYD
from domainlab.compos.pcr.request import RequestVAEBuilderCHW
from domainlab.compos.vae.utils_request_chain_builder import VAEChainNodeGetter
from domainlab.models.model_hduva import mk_hduva
from domainlab.utils.utils_cuda import get_device

from domainlab.algos.trainers.train_matchdg import TrainerMatchDG


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
        task.get_list_domains_tr_te(args.tr_d, args.te_d)
        device = get_device(args)
        node = VAEChainNodeGetter(request, args.topic_dim)()
        model = mk_hduva()(node,
                           zd_dim=args.zd_dim,
                           zy_dim=args.zy_dim,
                           zx_dim=args.zx_dim,
                           device=device,
                           topic_dim=args.topic_dim,
                           list_str_y=task.list_str_y,
                           gamma_d=args.gamma_d,
                           gamma_y=args.gamma_y,
                           beta_t=args.beta_t,
                           beta_x=args.beta_x,
                           beta_y=args.beta_y,
                           beta_d=args.beta_d)

        model = model.to(device)


        model_sel = MSelOracleVisitor(MSelValPerf(max_es=args.es))
        observer = ObVisitor(model_sel)

        trainer = TrainerMatchDG()
        trainer.init_business(model, task, observer, device, args)

        return trainer, model, observer, device
