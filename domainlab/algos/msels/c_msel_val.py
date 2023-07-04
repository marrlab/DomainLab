"""
Model Selection should be decoupled from
"""
from domainlab.algos.msels.c_msel_tr_loss import MSelTrLoss
from domainlab.utils.logger import Logger


class MSelValPerf(MSelTrLoss):
    """
    1. Model selection using validation performance
    2. Visitor pattern to trainer
    """
    def __init__(self, max_es):
        self.best_val_acc = 0.0
        super().__init__(max_es)  # construct self.tr_obs (observer)

    def update(self):
        """
        if the best model should be updated
        """
        flag = True
        if self.tr_obs.metric_val is None or self.tr_obs.str_msel == "loss_tr":
            return super().update()
        if self.tr_obs.metric_val["acc"] > self.best_val_acc:  # observer
            # different from loss, accuracy should be improved: the bigger the better
            self.best_val_acc = self.tr_obs.metric_val["acc"]
            # FIXME: only works for classification
            self.es_c = 0  # restore counter

        else:
            self.es_c += 1
            logger = Logger.get_logger()
            logger.debug("early stop counter: ", self.es_c)
            logger.debug(f"val acc:{self.tr_obs.metric_te['acc']}, "
                         f"best validation acc: {self.best_val_acc}")
            flag = False  # do not update best model

        return flag
