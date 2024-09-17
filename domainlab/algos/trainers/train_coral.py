"""
Deep CORAL: Correlation Alignment for Deep
Domain Adaptation
[au] Alexej, Xudong
"""
from domainlab.algos.trainers.mmd_base import TrainerMMDBase
from domainlab.utils.hyperparameter_retrieval import get_gamma_reg


class TrainerCoral(TrainerMMDBase):
    """
    cross domain MMD
    """
    def cross_domain_mmd(self, tuple_data_domains_batch):
        """
        domain-pairwise mmd
        """
        list_cross_domain_mmd = []
        list_domain_erm_loss = []
        num_domains = len(tuple_data_domains_batch)
        for ind_domain_a in range(num_domains):
            data_a, y_a, *_ = tuple_data_domains_batch[ind_domain_a]
            feat_a = self.model.extract_semantic_feat(data_a)
            list_domain_erm_loss.append(sum(self.get_model().cal_task_loss(data_a, y_a)))
            for ind_domain_b in range(ind_domain_a, num_domains):
                data_b, *_ = tuple_data_domains_batch[ind_domain_b]
                feat_b = self.model.extract_semantic_feat(data_b)
                mmd = self.mmd(feat_a, feat_b)
                list_cross_domain_mmd.append(sum(mmd))
        return list_domain_erm_loss, list_cross_domain_mmd

    def tr_epoch(self, epoch):
        list_loaders = list(self.dict_loader_tr.values())
        loaders_zip = zip(*list_loaders)
        self.model.train()
        self.model.convert4backpack()
        self.epo_loss_tr = 0

        for ind_batch, tuple_data_domains_batch in enumerate(loaders_zip):
            self.optimizer.zero_grad()
            list_domain_erm_loss, list_cross_domain_mmd = self.cross_domain_mmd(tuple_data_domains_batch)
            loss = sum(list_domain_erm_loss) + get_gamma_reg(self.aconf, self.name) * sum(list_cross_domain_mmd)
            loss.backward()
            self.optimizer.step()
            self.epo_loss_tr += loss.detach().item()
            self.after_batch(epoch, ind_batch)

        flag_stop = self.observer.update(epoch)  # notify observer
        return flag_stop
