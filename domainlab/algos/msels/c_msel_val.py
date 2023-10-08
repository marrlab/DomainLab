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
        self.sel_model_te_acc = 0.0
        self.best_te_metric = 0.0
        super().__init__(max_es)  # construct self.tr_obs (observer)

    def update(self):
        """
        if the best model should be updated
        """
        flag = True
        if self.tr_obs.metric_val is None or self.tr_obs.str_msel == "loss_tr":
            return super().update()
        metric = self.tr_obs.metric_val[self.tr_obs.str_metric4msel]
        if self.tr_obs.metric_te is not None:
            metric_te_current = self.tr_obs.metric_te[self.tr_obs.str_metric4msel]
            self.best_te_metric = max(self.best_te_metric, metric_te_current)

        if metric > self.best_val_acc:  # observer
            # different from loss, accuracy should be improved: the bigger the better
            self.best_val_acc = metric
            self.es_c = 0  # restore counter
            if self.tr_obs.metric_te is not None:
                metric_te_current = self.tr_obs.metric_te[self.tr_obs.str_metric4msel]
                self.sel_model_te_acc = metric_te_current

        else:
            self.es_c += 1
            logger = Logger.get_logger()
            logger.info(f"early stop counter: {self.es_c}")
            logger.info(f"val acc:{self.tr_obs.metric_val['acc']}, " +
                        f"best validation acc: {self.best_val_acc}, " +
                        f"corresponding to test acc: \
                        {self.sel_model_te_acc} / {self.best_te_metric}")
            flag = False  # do not update best model

        return flag
