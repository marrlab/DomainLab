"""
Wrapper for backpack and its extensions.
"""

import importlib

class BackpackWrapper:
    """
    Facilitates the use of backpack and its extensions.
    """
    def __init__(self):
        self.backpack = self._safe_import('backpack')
        self.extend = self._safe_import('backpack.extend')
        self.variance = self._safe_import('backpack.extensions', 'Variance')

    def _safe_import(self, module_name, attr=None):
        """
        Safely import a module and its attribute.
        :param module_name: The name of the module to be imported.
        """
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr) if attr else module
        except ImportError as e:
            print(f"Could not import {module_name}: {e}")
            return None

    def extend_loss_function(self, loss_function):
        """
        Extend the loss function with backpack's extend method.
        :param loss_function: The loss function to be extended.
        """
        if self.extend is not None:
            return self.extend(loss_function)
            
        raise ImportError("Backpack extension not available.")
        

    def apply_backpack(self, model, loss, extensions):
        """
        Apply backpack and its extensions to the model.

        :param model: The model to be extended with backpack.
        :param loss: The loss tensor to perform backward on.
        :param extensions: List of backpack extensions to be applied.
        """
        if self.backpack is not None:
            with self.backpack(*extensions):
                loss.backward(
                    inputs=list(model.parameters()), retain_graph=True, create_graph=True
                )
        else:
            raise ImportError("Backpack is not available.")
