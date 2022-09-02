from domainlab.algos.observers.b_obvisitor import ObVisitor
from domainlab.models.model_diva import ModelDIVA
from domainlab.utils.utils_classif import mk_dummy_label_list_str
from domainlab.compos.vae.utils_request_chain_builder import VAEChainNodeGetter
from domainlab.compos.pcr.request import RequestVAEBuilderCHW
from domainlab.algos.trainers.train_visitor import TrainerVisitor
from domainlab.compos.exp.exp_main import Exp
from domainlab.arg_parser import mk_parser_main
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.msels.c_msel import MSelTrLoss
from domainlab.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from domainlab.utils.utils_cuda import get_device


def test_trainer_diva():
    parser = mk_parser_main()
    argstr = "--te_d=rgb_31_119_180 --task=mnistcolor10 --aname=diva --bs=2 \
        --nocu"

    margs = parser.parse_args(argstr.split())
    margs.nname = "conv_bn_pool_2"
    y_dim = 10
    d_dim = 9
    list_str_y = mk_dummy_label_list_str("class", y_dim)
    list_str_d = mk_dummy_label_list_str("domain", d_dim)
    request = RequestVAEBuilderCHW(3, 28, 28, args=margs)

    node = VAEChainNodeGetter(request)()
    model = ModelDIVA(node, zd_dim=8, zy_dim=8, zx_dim=8, list_d_tr=list_str_d,
                      list_str_y=list_str_y, gamma_d=1.0, gamma_y=1.0,
                      beta_d=1.0, beta_y=1.0, beta_x=1.0)
    model_sel = MSelOracleVisitor(MSelTrLoss(max_es=margs.es))
    exp = Exp(margs)
    device = get_device(flag_no_cu=False)
    observer = ObVisitorCleanUp(ObVisitor(exp, model_sel, device))
    trainer = TrainerVisitor(model, task=exp.task, observer=observer, device=device, aconf=margs)
    trainer.before_tr()
    trainer.tr_epoch(0)
