"""
trainer for matchDG
"""
import copy

from domainlab.algos.compos.matchdg_ctr_erm import MatchCtrErm
from domainlab.algos.trainers.a_trainer import AbstractTrainer
from domainlab.utils.logger import Logger


class TrainerMatchDG(AbstractTrainer):
    """
    trainer for matchdg
    """
    def init_business(self, exp, task, model, observer, args, device):
        super().init_business(model, task, observer, device, args)
        self.exp = exp
        self.epo_loss_tr = None

        # different than model, ctr_model has no classification loss
        self.ctr_model = copy.deepcopy(model)
        self.ctr_model = self.ctr_model.to(device)
        self.erm = None
        self.args = self.aconf

    def before_tr(self):
        """
        configure trainer accoding to properties of task as well according to algorithm configs
        """
        # @FIXME: aconf and args should be separated
        # phase 1: contrastive learning
        ctr = MatchCtrErm(exp=self.exp,
                          task=self.task,
                          phi=self.ctr_model,
                          args=self.args,
                          device=self.device,
                          flag_erm=False,
                          opt=None)
        ctr.train()
        logger = Logger.get_logger()
        logger.info(f"Phase 1 finished: {ctr.ctr_mpath}")
        # phase 2: ERM, initialize object
        self.erm = MatchCtrErm(phi=self.model,
                               exp=self.exp,
                               task=self.task,
                               args=self.args,
                               device=self.device,
                               flag_erm=True,
                               opt=self.optimizer)

    def tr_epoch(self, epoch):
        self.model.train()
        self.erm.tr_epoch(epoch)
        self.epo_loss_tr = self.erm.epo_loss_tr
        flag_stop = self.observer.update(epoch)  # notify observer
        return flag_stop
