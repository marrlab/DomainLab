"""
trainer for matchDG
"""
import copy

from domainlab.algos.compos.matchdg_ctr_erm import MatchCtrErm
from domainlab.utils.logger import Logger


class TrainerMatchDG(MatchCtrErm):
    """
    trainer for matchdg
    """
    def init_business(self, exp, task, model, observer, args, device):
        self.exp = exp
        # different than model, ctr_model has no classification loss
        self.ctr_model = copy.deepcopy(model)
        self.ctr_model = self.ctr_model.to(device)
        self.erm = None
        super().init_business(self.ctr_model, task, observer, device, args, flag_erm=False)

    def before_tr(self):
        """
        configure trainer accoding to properties of task as well according to algorithm configs
        """
        # phase 1: contrastive learning
        # different than phase 2, ctr_model has no classification loss
        for epoch in range(self.aconf.epochs_ctr):
            self.tr_epoch(epoch)
        logger = Logger.get_logger()
        logger.info(f"Phase 1 finished: {self.model_path_ctr}")
        # phase 2: ERM, initialize object
        self.observer.reset()
        self.aconf.epos = self.aconf.epos - self.aconf.epochs_ctr
        super().init_business(self.model, self.task, self.observer, self.device,
                              self.aconf, flag_erm=True)
