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
        self.inner_trainer.extend(self._decoratee)
        inner_model = copy.deepcopy(self.model)
        self.inner_trainer.init_business(
            inner_model, copy.deepcopy(self.task), self.observer, self.device, self.aconf,
            flag_accept=False)
        self.prepare_ziped_loader()

    def prepare_ziped_loader(self):
        """
        create virtual source and target domain
        """
        list_dsets = list(self.task.dict_dset_tr.values())
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
        # s means source, t means target
        for ind_batch, (tensor_x_s, vec_y_s, vec_d_s, others_s,
                        tensor_x_t, vec_y_t, vec_d_t, *_) \
                in enumerate(self.loader_tr_source_target):

            tensor_x_s, vec_y_s, vec_d_s = \
                tensor_x_s.to(self.device), vec_y_s.to(self.device), vec_d_s.to(self.device)

            tensor_x_t, vec_y_t, vec_d_t = \
                tensor_x_t.to(self.device), vec_y_t.to(self.device), vec_d_t.to(self.device)

            self.optimizer.zero_grad()

            self.inner_trainer.model.load_state_dict(self.model.state_dict())
            # update inner_model
            self.inner_trainer.before_epoch()  # set model to train mode
            self.inner_trainer.reset()  # force optimizer to re-initialize
            self.inner_trainer.tr_batch(
                tensor_x_s, vec_y_s, vec_d_s, others_s, ind_batch, epoch)
            # inner_model has now accumulated gradients Gi
            # with parameters theta_i - lr * G_i where i index batch

            loss_look_forward = self.inner_trainer.model.cal_task_loss(tensor_x_t, vec_y_t)
            loss_source_task = self.model.cal_task_loss(tensor_x_s, vec_y_s)
            list_source_reg_tr, list_source_mu_tr = self.cal_reg_loss(tensor_x_s, vec_y_s, vec_d_s, others_s)
            # call cal_reg_loss from decoratee
            # super()._cal_reg_loss returns [],[],
            # since mldg's reg loss is on target domain,
            # no other trainer except hyperscheduler could decorate it unless we use state pattern
            # in the future to control source and target domain loader behavior
            source_reg_tr = self.model.inner_product(list_source_reg_tr, list_source_mu_tr)
            # self.aconf.gamma_reg * loss_look_forward.sum()
            loss = loss_source_task.sum() + source_reg_tr.sum() +\
                    self.aconf.gamma_reg * loss_look_forward.sum()
            #
            loss.backward()
            # optimizer only optimize parameters of self.model, not inner_model
            self.optimizer.step()
            self.epo_loss_tr += loss.detach().item()
            self.after_batch(epoch, ind_batch)
        flag_stop = self.observer.update(epoch)  # notify observer
        return flag_stop
