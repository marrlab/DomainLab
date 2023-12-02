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
    def before_tr(self):
        """
        configure trainer accoding to properties of task as well according to algorithm configs
        """
        logger = Logger.get_logger()
        logger.info("\n\nPhase 1 start: contractive alignment without task loss: \n\n")
        # phase 1: contrastive learning
        # different than phase 2, ctr_model has no classification loss
