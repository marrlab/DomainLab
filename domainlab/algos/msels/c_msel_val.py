"""
Model Selection should be decoupled from
"""
from domainlab.algos.msels.a_model_sel import AMSel


class MSelValPerf(AMSel):
    """
    1. Model selection using validation performance
    2. Visitor pattern to trainer
    """
    def __init__(self, max_es):
        self.es_c = 0
        self.max_es = max_es
        self.best_val_acc = 0
        super().__init__()  # construct self.tr_obs (observer)

    def update(self):
        """
        if the best model should be updated
        """
        flag = True
        if self.tr_obs.metric_val["acc"] > self.best_val_acc:  # observer
            self.best_val_acc = self.tr_obs.metric_te["acc"]
            # FIXME: only works for classification
            self.es_c = 0  # restore counter

        else:
            self.es_c += 1
            print("early stop counter: ", self.es_c)
            print(f"val acc:{self.tr_obs.metric_te['acc']}, best validation acc: {self.best_val_acc}")
            flag = False  # do not update best model

        return flag

    def if_stop(self):
        """
        if should early stop
        """
        return self.es_c > self.max_es
