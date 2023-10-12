"""
Builder pattern to build different component for experiment with DIVA
"""
from domainlab.algos.a_algo_builder import NodeAlgoBuilder
from domainlab.algos.msels.c_msel_val import MSelValPerf
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.b_obvisitor import ObVisitor
from domainlab.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from domainlab.algos.observers.c_obvisitor_gen import ObVisitorGen
from domainlab.algos.trainers.zoo_trainer import TrainerChainNodeGetter

from domainlab.compos.pcr.request import RequestVAEBuilderCHW
from domainlab.compos.vae.utils_request_chain_builder import VAEChainNodeGetter
from domainlab.models.model_diva import mk_diva
from domainlab.utils.utils_cuda import get_device


class NodeAlgoBuilderDIVA(NodeAlgoBuilder):
    """
    Builder pattern to build different component for experiment with DIVA
    """
    def get_trainer(self, args):
        """
        chain of responsibility pattern for fetching trainer from dictionary
        """
        trainer = TrainerChainNodeGetter(args)(default="hyperscheduler")
        return trainer

    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        request = RequestVAEBuilderCHW(
            task.isize.c, task.isize.h, task.isize.w, args)
        node = VAEChainNodeGetter(request)()
        model = mk_diva(str_diva_multiplier_type=args.str_diva_multiplier_type)(node,
                          zd_dim=args.zd_dim,
                          zy_dim=args.zy_dim,
                          zx_dim=args.zx_dim,
                          list_str_y=task.list_str_y,
                          list_d_tr=task.list_domain_tr,
                          gamma_d=args.gamma_d,
                          gamma_y=args.gamma_y,
                          beta_x=args.beta_x,
                          beta_y=args.beta_y,
                          beta_d=args.beta_d)
        device = get_device(args)
        model_sel = MSelOracleVisitor(MSelValPerf(max_es=args.es))
        if not args.gen:
            observer = ObVisitorCleanUp(
                ObVisitor(model_sel,
                          device,
                          exp=exp))
        else:
            observer = ObVisitorCleanUp(
                ObVisitorGen(model_sel,
                             device,
                             exp=exp))
        trainer = self.get_trainer(args)
        return trainer, model, observer, device
