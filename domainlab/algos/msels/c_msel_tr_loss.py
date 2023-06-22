"""
Model Selection should be decoupled from
"""
import math
from domainlab.algos.msels.a_model_sel import AMSel


class MSelTrLoss(AMSel):
    """
    1. Model selection using sum of loss across training domains
    2. Visitor pattern to trainer
    """
    def __init__(self, max_es):
        self.best_loss = float("inf")
        self.es_c = 0
        self.max_es = max_es
        super().__init__()

    def update(self):
        """
        if the best model should be updated
        """
        loss = self.trainer.epo_loss_tr   # @FIXME
        assert loss is not None
        assert not math.isnan(loss)
        flag = True
        if loss < self.best_loss:
            self.es_c = 0  # restore counter
            self.best_loss = loss
        else:
            self.es_c += 1
            print("early stop counter: ", self.es_c)
            print(f"loss:{loss}, best loss: {self.best_loss}")
            flag = False  # do not update best model
        return flag

    def if_stop(self):
        """
        if should early stop
        """
        return self.es_c > self.max_es
