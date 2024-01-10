"""
Emperical risk minimization
"""
from domainlab.models.a_model_classif import AModelClassif
from domainlab.utils.override_interface import override_interface
from domainlab.compos.nn_zoo.nn import LayerId


def mk_erm(parent_class=AModelClassif):
    """
    Instantiate a Deepall (ERM) model

    Details:
        Creates a model, which trains a neural network via standard empirical risk minimization
        (ERM). The fact that the training data stems from different domains is neglected, as all
        domains are pooled together during training.

    Args:
        parent_class (AModel, optional):
            Class object determining the task type. Defaults to AModelClassif.

    Returns:
        ModelERM: model inheriting from parent class

    Input Parameters:
        custom neural network, the output dimension must be the number of labels

    Usage:
        For a concrete example, see:
        https://github.com/marrlab/DomainLab/blob/tests/test_mk_exp_erm.py
    """

    class ModelERM(parent_class):
        """
        anonymous
        """
        def __init__(self, net=None, net_feat=None, net_classifier=None, list_str_y=None):
            if net_feat is None and net_classifier is None and net is not None:
                net_feat = net
                net_classifier = LayerId()
                dim_y = list(net.modules())[-1].out_features
            elif net_classifier is not None:
                dim_y = list(net_classifier.modules())[-1].out_features
            else:
                raise RuntimeError("specify either a whole network for classification or separate \
                        feature and classifier")
            if list_str_y is None:
                list_str_y = [f"class{i}" for i in range(dim_y)]
            super().__init__(list_str_y)
            self._net_classifier = net_classifier
            self._net_invar_feat = net_feat
    return ModelERM
