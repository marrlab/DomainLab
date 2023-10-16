"""
Model Selection should be decoupled from
"""
from domainlab.algos.msels.c_msel_val import MSelValPerf
from domainlab.utils.logger import Logger


class MSelValPerfTopK(MSelValPerf):
    """
    1. Model selection using validation performance
    2. Visitor pattern to trainer
    """
    def __init__(self, max_es, top_k=2):
        super().__init__(max_es)  # construct self.tr_obs (observer)
        self.top_k = top_k
        self.list_top_k_acc = [0.0 for _ in range(top_k)]

    def update(self, clear_counter=False):
        """
        if the best model should be updated
        """
        flag_super = super().update(clear_counter)
        metric_val_current = \
            self.tr_obs.metric_val[self.tr_obs.str_metric4msel]
        acc_min = min(self.list_top_k_acc)
        if metric_val_current > acc_min:
            # overwrite
            self.es_c = 0  # restore counter
            ind = self.list_top_k_acc.index(acc_min)
            self.list_top_k_acc[ind] = metric_val_current
            # overwrite
            metric_te_current = \
                self.tr_obs.metric_te[self.tr_obs.str_metric4msel]
            self._sel_model_te_acc = metric_te_current
            logger = Logger.get_logger()
            logger.info(f"top k validation acc updated: \
                        {self.list_top_k_acc}, \
                        overwriting counter, selected test acc")
            return True
        return flag_super
