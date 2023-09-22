"""
Model Selection should be decoupled from
"""
from domainlab.algos.msels.a_model_sel import AMSel


class MSelBang(AMSel):
    """
    1. Model selection using validation performance
    2. Visitor pattern to trainer
    """
    def __init__(self, max_es):
        self.best_val_acc = 0.0

    def if_stop(self):
        return False

    def update(self):
        """
        if the best model should be updated
        """
        return True
