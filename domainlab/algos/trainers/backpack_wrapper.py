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


