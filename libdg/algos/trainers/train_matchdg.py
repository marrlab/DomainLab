from libdg.algos.compos.matchdg_ctr_erm import MatchCtrErm
from libdg.algos.trainers.a_trainer import AbstractTrainer


class TrainerMatchDG(AbstractTrainer):
    def __init__(self, exp, task, ctr_model, model, observer, args, device):
        super().__init__(model, task, observer, device, args)
        self.exp = exp
        self.epo_loss_tr = None
        self.ctr_model = ctr_model
        self.args = self.aconf

    def before_tr(self):
        """
        configure trainer accoding to properties of task as well according to algorithm configs
        """
        # FIXME: aconf and args should be separated
        # phase 1: contrastive learning
        ctr = MatchCtrErm(exp=self.exp,
                          task=self.task,
                          phi=self.ctr_model,
                          args=self.args,
                          device=self.device,
                          flag_erm=False)
        ctr.train()
        print("Phase 1 finished: ", ctr.ctr_mpath)
        # phase 2: ERM, initialize object
        self.erm = MatchCtrErm(phi=self.model,
                               exp=self.exp,
                               task=self.task,
                               args=self.args,
                               device=self.device,
                               flag_erm=True)

    def tr_epoch(self, epoch):
        self.model.train()
        self.erm.tr_epoch(epoch)
        self.epo_loss_tr = self.erm.epo_loss_tr
        flag_stop = self.observer.update(epoch)  # notify observer
        return flag_stop
