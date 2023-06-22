"""
Meta Learning Domain Generalization
"""
import copy
import random
from torch.utils.data.dataset import ConcatDataset

from domainlab.algos.trainers.a_trainer import AbstractTrainer
from domainlab.algos.trainers.train_basic import TrainerBasic
from domainlab.tasks.utils_task import mk_loader
from domainlab.tasks.utils_task_dset import DsetZip


class TrainerMLDG(AbstractTrainer):
    """
    basic trainer
    """
    def before_tr(self):
        """
        check the performance of randomly initialized weight
        """
        self.model.evaluate(self.loader_te, self.device)
        self.inner_trainer = TrainerBasic()
        self.inner_trainer.init_business(
            self.model, self.task, self.observer, self.device, self.aconf,
            flag_accept=False)
        self.prepare_ziped_loader()

    def prepare_ziped_loader(self):
        """
        create virtual source and target domain
        """
        list_dsets = list(self.task.dict_dset.values())
        num_domains = len(list_dsets)
        ind_target_domain = random.randrange(num_domains)
        tuple_dsets_source = tuple(
            list_dsets[ind] for ind in range(num_domains) if ind != ind_target_domain)
        ddset_source = ConcatDataset(tuple_dsets_source)
        ddset_target = list_dsets[ind_target_domain]
        ddset_mix = DsetZip(ddset_source, ddset_target)
        self.loader_tr_source_target = mk_loader(ddset_mix, self.aconf.bs)

    def tr_epoch(self, epoch):
        self.model.train()
        self.epo_loss_tr = 0
        self.prepare_ziped_loader()
        for ind_batch, (tensor_x_s, vec_y_s, vec_d_s, others_s,
                        tensor_x_t, vec_y_t, vec_d_t, *_) \
                in enumerate(self.loader_tr_source_target):

            tensor_x_s, vec_y_s, vec_d_s = \
                tensor_x_s.to(self.device), vec_y_s.to(self.device), vec_d_s.to(self.device)

            tensor_x_t, vec_y_t, vec_d_t = \
                tensor_x_t.to(self.device), vec_y_t.to(self.device), vec_d_t.to(self.device)

            self.optimizer.zero_grad()

            inner_net = copy.deepcopy(self.model)
            self.inner_trainer.model = inner_net   # FORCE replace model
            self.inner_trainer.train_batch(
                tensor_x_s, vec_y_s, vec_d_s, others_s)  # update inner_net

            # DomainBed:
            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            # for p_tgt, p_src in zip(self.model.parameters(),
            #                         inner_net.parameters()):
            #    if p_src.grad is not None:
            #         p_tgt.grad.data.add_(p_src.grad.data / num_mb)
            loss_look_forward = inner_net.cal_task_loss(tensor_x_t, vec_y_t)
            # DomainBed: instead of backward, do explicit gradient update
            # loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            # grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(), allow_unused=True)
            # for p, g_j in zip(self.model.parameters(), grad_inner_j):
            #     if g_j is not None:
            #         p.grad.data.add_(
            #             self.hparams['mldg_beta'] * g_j.data / num_mb)
            loss_source = self.model.cal_loss(tensor_x_s, vec_y_s, vec_d_s, others_s)
            loss = loss_source.sum() + self.aconf.gamma_reg * loss_look_forward.sum()
            loss.backward()
            self.optimizer.step()
            self.epo_loss_tr += loss.detach().item()
            self.after_batch(epoch, ind_batch)
        flag_stop = self.observer.update(epoch)  # notify observer
        return flag_stop
