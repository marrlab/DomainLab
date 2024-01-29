import importlib

class BackpackWrapper:
    def __init__(self):
        self.backpack = self._safe_import('backpack')
        self.extend = self._safe_import('backpack.extend')
        self.Variance = self._safe_import('backpack.extensions', 'Variance')

    def _safe_import(self, module_name, attr=None):
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr) if attr else module
        except ImportError as e:
            print(f"Could not import {module_name}: {e}")
            return None

    def extend_loss_function(self, loss_function):
        if self.extend is not None:
            return self.extend(loss_function)
        else:
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


