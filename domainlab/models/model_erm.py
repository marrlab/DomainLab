"""
Emperical risk minimization
"""
from domainlab.compos.nn_zoo.nn import LayerId
from domainlab.models.a_model_classif import AModelClassif
from domainlab.utils.override_interface import override_interface


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

        def __init__(self, net=None, net_feat=None, **kwargs):
            if net is not None:
                net_feat = net
                kwargs["net_classifier"] = LayerId()
            super().__init__(kwargs)
            self._net_invar_feat = net_feat
    return ModelERM
