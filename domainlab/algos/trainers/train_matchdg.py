"""
trainer for matchDG
"""
from torch import optim
from domainlab.algos.compos.matchdg_ctr_erm import MatchCtrErm
from domainlab.algos.trainers.a_trainer import AbstractTrainer


class TrainerMatchDG(AbstractTrainer):
    """
    trainer for matchdg
    """
    def init_business(self, exp, task, ctr_model, model, observer, args, device):
        super().init_business(model, task, observer, device, args)
        self.exp = exp
        self.epo_loss_tr = None
        self.ctr_model = ctr_model
        self.erm = None
        self.args = self.aconf

    def get_opt_sgd(self):
        opt = optim.SGD([{'params': filter(
                         lambda p: p.requires_grad,
                         self.phi.parameters())}, ],
                        lr=self.args.lr, weight_decay=5e-4,
                        momentum=0.9, nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=25)
        return opt

    def before_tr(self):
        """
        configure trainer accoding to properties of task as well according to algorithm configs
        """
        # @FIXME: aconf and args should be separated
        # phase 1: contrastive learning
        opt = self.get_opt_sgd()
        ctr = MatchCtrErm(exp=self.exp,
                          task=self.task,
                          phi=self.ctr_model,
                          args=self.args,
                          device=self.device,
                          flag_erm=False,
                          opt=opt)
        ctr.train()
        print("Phase 1 finished: ", ctr.ctr_mpath)
        # phase 2: ERM, initialize object
        self.erm = MatchCtrErm(phi=self.model,
                               exp=self.exp,
                               task=self.task,
                               args=self.args,
                               device=self.device,
                               flag_erm=True,
                               opt=opt)

    def tr_epoch(self, epoch):
        self.model.train()
        self.erm.tr_epoch(epoch)
        self.epo_loss_tr = self.erm.epo_loss_tr
        flag_stop = self.observer.update(epoch)  # notify observer
        return flag_stop
