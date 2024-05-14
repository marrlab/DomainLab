"""
Emperical risk minimization
"""
from domainlab.compos.nn_zoo.nn import LayerId
from domainlab.models.a_model_classif import AModelClassif
from domainlab.utils.override_interface import override_interface
import traceback

try:
    from backpack import extend
except ImportError as e:
    print(f"Failed to import 'extend' from backpack: {e}")
    extend = None  # Ensure extend is defined to avoid NameError later in the code



def mk_erm(parent_class=AModelClassif, **kwargs):
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

        def __init__(self, net=None, net_feat=None):
            if net is not None:
                net_feat = net
                kwargs["net_classifier"] = LayerId()
            super().__init__(**kwargs)
            self._net_invar_feat = net_feat

        def convert4backpack(self):
            print("Extending model components...")
            if extend is not None:
                try:
                    if hasattr(self._net_invar_feat, 'parameters'):
                        print("Net features before extend:", self._net_invar_feat)
                    self._net_invar_feat = extend(self._net_invar_feat, use_converter=True)
                except Exception as e:
                    print("An error occurred:", e)
                    traceback.print_exc()
                
                if hasattr(self.net_classifier, 'parameters'):
                    print("Net classifier before extend:", self.net_classifier)
                self.net_classifier = extend(self.net_classifier, use_converter=True)
            else:
                print("Backpack's extend function is not available.")


    
        def hyper_update(self, epoch, fun_scheduler):
            """hyper_update.

            :param epoch:
            :param fun_scheduler:
            """
            pass

        def hyper_init(self, functor_scheduler, trainer=None):
            """
            initiate a scheduler object via class name and things inside this model

            :param functor_scheduler: the class name of the scheduler
            """
            return functor_scheduler(
                trainer=trainer
            )

    return ModelERM
