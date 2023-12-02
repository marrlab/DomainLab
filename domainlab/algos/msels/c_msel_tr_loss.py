"""
Model Selection should be decoupled from
"""
import math
from domainlab.algos.msels.a_model_sel import AMSel
from domainlab.utils.logger import Logger


class MSelTrLoss(AMSel):
    """
    1. Model selection using sum of loss across training domains
    2. Visitor pattern to trainer
    """
    def __init__(self, max_es):
        super().__init__()
        # NOTE: super() must come first otherwise it will overwrite existing
        # values!
        self.reset()
        self._max_es = max_es

    def reset(self):
        self.best_loss = float("inf")
        self.es_c = 0

    @property
    def max_es(self):
        return self._max_es

    def update(self, clear_counter=False):
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
            logger = Logger.get_logger()
            logger.info(f"early stop counter: {self.es_c}")
            logger.info(f"loss:{loss}, best loss: {self.best_loss}")
            flag = False  # do not update best model
            if clear_counter:
                logger.info("clearing counter")
                self.es_c = 0
        return flag

    def if_stop(self):
        """
        if should early stop
        """
        return self.es_c > self.max_es
