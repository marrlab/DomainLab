"""
Model Selection should be decoupled from
"""
from domainlab.algos.msels.a_model_sel import AMSel


class MSelOracleVisitor(AMSel):
    """
    save best out-of-domain test acc model, but do not affect
    how the final model is selected
    """
    def __init__(self, msel):
        """
        Decorator pattern
        """
        super().__init__()
        self.best_oracle_acc = 0
        self.msel = msel

    def update(self):
        """
        if the best model should be updated
        """
        self.tr_obs.exp.visitor.save(self.trainer.model, "epoch")
        if self.tr_obs.metric_te["acc"] > self.best_oracle_acc:
            self.best_oracle_acc = self.tr_obs.metric_te["acc"]  # FIXME: only works for classification
            self.tr_obs.exp.visitor.save(self.trainer.model, "oracle")
            print("oracle model saved")
        return self.msel.update()

    def if_stop(self):
        """
        if should early stop
        oracle model selection does not intervene how models get selected
        by the innermost model selection
        """
        return self.msel.if_stop()

    def accept(self, trainer, tr_obs):
        self.msel.accept(trainer, tr_obs)
        super().accept(trainer, tr_obs)
