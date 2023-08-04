"""
make an experiment
"""
from domainlab.arg_parser import mk_parser_main
from domainlab.compos.exp.exp_main import Exp

def mk_exp(task, test_domain, batchsize):
    str_arg = f"--te_d={test_domain} --bs={batchsize}"
    parser = mk_parser_main()
    conf = parser.parse_args(str_arg.split())
    exp = Exp(conf, task)
    return exp



def test():

    from torchvision import models as torchvisionmodels
    from torchvision.models import ResNet50_Weights
    from domainlab.tasks.dset import mk_task_dset
    from domainlab.models.model_deep_all import mk_deepall
    from domainlab.algos.trainers.trainer_mldg import TrainerMLDG

    from domainlab.compos.nn_zoo.nn import LayerId
    from domainlab.compos.nn_zoo.nn_torchvision import NetTorchVisionBase

    task = mk_task_dset
    exp = mk_exp(task, test_domain=["0", "1", "2"], batchsize=32)
    # model_sel = MSelValPerf(max_es=1)
    # observer = ObVisitor(exp, model_sel, device=torch.device("gpu"))
    # code backbone
    dim_y = 10
    backbone = torchvisionmodels.resnet.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_final_in = backbone.fc.in_features
    backbone.fc = nn.Linear(num_final_in, dim_y)
    ## make model
    model = mk_deepall()(backbone)
    ## make trainer for model
    trainer = TrainerMLDG()
    trainer.init_business(model, task, observer, device, args)
