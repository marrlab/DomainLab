"""
Model Selection should be decoupled from
"""
from domainlab.algos.msels.a_model_sel import AMSel
from domainlab.utils.logger import Logger


class MSelOracleVisitor(AMSel):
    """
    save best out-of-domain test acc model, but do not affect
    how the final model is selected
    """
    def __init__(self, msel=None):
        """
        Decorator pattern
        """
        super().__init__()
        self.best_oracle_acc = 0
        self.msel = msel

    @property
    def best_val_acc(self):
        """
        decoratee best val acc
        """
        if self.msel is not None and hasattr(self.msel, "best_val_acc"):
            return self.msel.best_val_acc
        return -1

    @property
    def best_te_metric(self):
        """
        decoratee best test metric
        """
        if self.msel is not None and hasattr(self.msel, "best_te_metric"):
            return self.msel.best_te_metric
        return -1

    def update(self, clear_counter=False):
        """
        if the best model should be updated
        """
        self.tr_obs.exp.visitor.save(self.trainer.model, "epoch")
        flag = False
        metric = self.tr_obs.metric_te[self.tr_obs.str_metric4msel]
        if metric > self.best_oracle_acc:
            self.best_oracle_acc = metric
            if self.msel is not None:
                self.tr_obs.exp.visitor.save(self.trainer.model, "oracle")
            else:
                self.tr_obs.exp.visitor.save(self.trainer.model)
            logger = Logger.get_logger()
            logger.info("new oracle model saved")
            flag = True
        if self.msel is not None:
            return self.msel.update(clear_counter)
        return flag

    def if_stop(self):
        """
        if should early stop
        oracle model selection does not intervene how models get selected
        by the innermost model selection
        """
        if self.msel is not None:
            return self.msel.if_stop()
        return False

    def accept(self, trainer, tr_obs):
        if self.msel is not None:
            self.msel.accept(trainer, tr_obs)
        super().accept(trainer, tr_obs)
