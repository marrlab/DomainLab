from domainlab.algos.a_algo_builder import NodeAlgoBuilder
from domainlab.algos.trainers.train_basic import TrainerBasic
from domainlab.algos.msels.c_msel import MSelTrLoss
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.b_obvisitor import ObVisitor
from domainlab.utils.utils_cuda import get_device
from domainlab.compos.nn_alex import Alex4DeepAll
from domainlab.models.model_deep_all import ModelDeepAll


class NodeAlgoBuilderCustom(NodeAlgoBuilder):
    """
    When you implement your own algorithm you have to inherit
    the NodeAlgoBuilder interface and override the init_business
    function
    """

    def init_business(self, exp):
        """
        return trainer
        """
        task = exp.task
        args = exp.args
        device = get_device(args.nocu)

        # observer is responsible for model selection (e.g. early stop)
        # and logging epoch-wise performance
        observer = ObVisitor(exp=exp,
                             model_sel=MSelTrLoss(max_es=args.es),
                             device=device)

        # One could define their own model with custom loss, but the model
        # must conform with the parent class of ModelDeepAll
        net = Alex4DeepAll(flag_pretrain=True, dim_y=task.dim_y)
        model = ModelDeepAll(net, list_str_y=task.list_str_y)

        # trainer is responsible for directing the data flow, which contains
        # all the objects: model, task, observer
        trainer = TrainerBasic(model, task, observer, device, args)
        return trainer


def get_node_na():
    """In your custom python file, this function has to be implemented
    to return the custom algorithm builder"""
    return NodeAlgoBuilderCustom
