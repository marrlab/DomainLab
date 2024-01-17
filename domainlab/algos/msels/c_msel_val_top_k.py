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
        metric_val_current = self.tr_obs.metric_val[self.tr_obs.str_metric4msel]
        acc_min = min(self.list_top_k_acc)
        if metric_val_current > acc_min:
            # overwrite
            logger = Logger.get_logger()
            logger.info(
                f"top k validation acc: {self.list_top_k_acc} \
                        overwriting/reset  counter"
            )
            self.es_c = 0  # restore counter
            ind = self.list_top_k_acc.index(acc_min)
            # avoid having identical values
            if metric_val_current not in self.list_top_k_acc:
                self.list_top_k_acc[ind] = metric_val_current
                logger.info(
                    f"top k validation acc updated: \
                            {self.list_top_k_acc}"
                )
                # overwrite to ensure consistency
                # issue #569: initially self.list_top_k_acc will be [xx, 0] and it does not matter since 0 will be overwriten by second epoch validation acc.
                # actually, after epoch 1, most often, sefl._best_val_acc will be the higher value of self.list_top_k_acc will overwriten by min(self.list_top_k_acc)
                logger.info(
                    f"top-2 val sel: overwriting best val acc from {self._best_val_acc} to "
                    f"minimum of {self.list_top_k_acc} which is {min(self.list_top_k_acc)} "
                    f"to ensure consistency"
                )
                self._best_val_acc = min(self.list_top_k_acc)
            # overwrite test acc, this does not depend on if val top-k acc has been overwritten or not
            metric_te_current = self.tr_obs.metric_te[self.tr_obs.str_metric4msel]
            if self._sel_model_te_acc != metric_te_current:
                # this can only happen if the validation acc has decreased and current val acc is only bigger than min(self.list_top_k_acc} but lower than max(self.list_top_k_acc)
                logger.info(
                    f"top-2 val sel: overwriting selected model test acc from "
                    f"{self._sel_model_te_acc} to {metric_te_current} to ensure consistency"
                )
            self._sel_model_te_acc = metric_te_current
            return True
        return flag_super
