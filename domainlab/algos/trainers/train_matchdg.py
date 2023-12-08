"""
trainer matchdg
"""
import torch

from domainlab.algos.trainers.a_trainer import AbstractTrainer
from domainlab.algos.trainers.compos.matchdg_utils import \
get_base_domain_size4match_dg
from domainlab.algos.trainers.compos.matchdg_match import MatchPair
from domainlab.algos.trainers.compos.matchdg_utils import (dist_cosine_agg,
                                                  dist_pairwise_cosine)
from domainlab.utils.logger import Logger
from domainlab.tasks.utils_task_dset import DsetIndDecorator4XYD


class TrainerMatchDG(AbstractTrainer):
    """
    Contrastive Learning
    """
    def dset_decoration_args_algo(self, args, ddset):
        ddset = DsetIndDecorator4XYD(ddset)
        return ddset

    def init_business(self, model, task, observer, device, aconf, flag_accept=True, flag_erm=False):
        """
        initialize member objects
        """
        super().init_business(model, task, observer, device, aconf, flag_accept)
        # use the same batch size for match tensor
        # so that order is kept!
        self.base_domain_size = get_base_domain_size4match_dg(self.task)
        self.epo_loss_tr = 0
        self.flag_erm = flag_erm
        self.lambda_ctr = self.aconf.gamma_reg
        self.mk_match_tensor(epoch=0)
        self.flag_stop = False
        self.tuple_tensor_ref_domain2each_y = None
        self.tuple_tensor_refdomain2each = None

    def tr_epoch(self, epoch):
        """
        # data in one batch comes from two sources: one part from loader,
        # the other part from match tensor
        """
        self.model.train()
        self.epo_loss_tr = 0
        logger = Logger.get_logger()
        # update match tensor
        if (epoch + 1) % self.aconf.epos_per_match_update == 0:
            self.mk_match_tensor(epoch)

        inds_shuffle = torch.randperm(self.tensor_ref_domain2each_domain_x.size(0))
        # NOTE: match tensor size: N(ref domain size) * #(train domains) * (image size: c*h*w)
        # self.tensor_ref_domain2each_domain_x[inds_shuffle]
        # shuffles the match tensor at the first dimension
        self.tuple_tensor_refdomain2each = torch.split(
            self.tensor_ref_domain2each_domain_x[inds_shuffle],
            self.aconf.bs, dim=0)
        # Splits the tensor into chunks.
        # Each chunk is a view of the original tensor of batch size self.aconf.bs
        # return is a tuple of the splited chunks
        self.tuple_tensor_ref_domain2each_y = torch.split(
            self.tensor_ref_domain2each_domain_y[inds_shuffle],
            self.aconf.bs, dim=0)
        logger.info(f"number of batches in match tensor: {len(self.tuple_tensor_refdomain2each)}")
        logger.info(f"single batch match tensor size: {self.tuple_tensor_refdomain2each[0].shape}")

        for batch_idx, (x_e, y_e, d_e, *others) in enumerate(self.loader_tr):
            # random loader with same batch size as the match tensor loader
            # the 4th output of self.loader is not used at all,
            # is only used for creating the match tensor
            self.tr_batch(epoch, batch_idx, x_e, y_e, d_e, others)
            if self.flag_stop is True:
                logger.info("ref/base domain vs each domain match \
                            traversed one sweep, starting new epoch")
                break
        if epoch < self.aconf.epochs_ctr:
            logger.info("\n\nPhase 0 continue\n\n")
            return False
        self.flag_erm = True
        flag_stop = self.observer.update(epoch)  # notify observer
        return flag_stop

    def tr_batch(self, epoch, batch_idx, x_e, y_e, d_e, others=None):
        """
        update network for each batch
        """
        self.optimizer.zero_grad()
        x_e = x_e.to(self.device)  # 64 * 1 * 224 * 224
        # y_e_scalar = torch.argmax(y_e, dim=1).to(self.device)
        y_e = y_e.to(self.device)
        # d_e = torch.argmax(d_e, dim=1).numpy()
        d_e = d_e.to(self.device)
        # for each batch, the list loss is re-initialized

        # CTR (contrastive) loss for CTR/ERM phase are different
        list_batch_loss_ctr = []
        # for a single batch,  loss need to be
        # aggregated across different combinations of domains.
        # Defining a leaf node can cause problem by loss_ctr += xxx,
        # a list with python built-in "sum" can aggregate
        # these losses within one batch

        if self.flag_erm:
            list_loss_reg_rand, list_mu_reg = self.decoratee.cal_reg_loss(x_e, y_e, d_e, others)
            loss_reg = self.model.inner_product(list_loss_reg_rand, list_mu_reg)
            loss_task_rand = self.model.cal_task_loss(x_e, y_e)
            # loss_erm_rnd_loader, *_ = self.model.cal_loss(x_e, y_e, d_e, others)
            loss_erm_rnd_loader = loss_reg + loss_task_rand

        num_batches = len(self.tuple_tensor_refdomain2each)

        if batch_idx >= num_batches:
            logger = Logger.get_logger()
            logger.info("ref/base domain vs each domain match"
                        "traversed one sweep, starting new epoch")
            self.flag_stop = True
            return

        curr_batch_size = self.tuple_tensor_refdomain2each[batch_idx].shape[0]

        batch_tensor_ref_domain2each = self.tuple_tensor_refdomain2each[batch_idx].to(self.device)
        # make order 5 tensor: (ref_domain, domain, channel, img_h, img_w)
        # with first dimension as batch size

        # clamp the first two dimensions so the model network could map image to feature
        batch_tensor_ref_domain2each = batch_tensor_ref_domain2each.view(
            batch_tensor_ref_domain2each.shape[0]*batch_tensor_ref_domain2each.shape[1],
            batch_tensor_ref_domain2each.shape[2],   # channel
            batch_tensor_ref_domain2each.shape[3],   # img_h
            batch_tensor_ref_domain2each.shape[4])   # img_w
        # now batch_tensor_ref_domain2each first dim will not be batch_size!
        # batch_tensor_ref_domain2each.shape torch.Size([40, channel, 224, 224])

        batch_feat_ref_domain2each = self.model.extract_semantic_feat(
            batch_tensor_ref_domain2each)
        # batch_feat_ref_domain2each.shape torch.Size[40, 512]
        # torch.sum(torch.isnan(batch_tensor_ref_domain2each))
        # assert not torch.sum(torch.isnan(batch_feat_ref_domain2each))
        flag_isnan = torch.any(torch.isnan(batch_feat_ref_domain2each))
        if flag_isnan:
            logger = Logger.get_logger()
            logger.info(batch_tensor_ref_domain2each)
            raise RuntimeError("batch_feat_ref_domain2each NAN! is learning rate too big or"
                               "hyper-parameter tau not set appropriately?")

        # for contrastive training phase,
        # the last layer of the model is replaced with identity

        batch_ref_domain2each_y = self.tuple_tensor_ref_domain2each_y[batch_idx].to(self.device)
        batch_ref_domain2each_y = batch_ref_domain2each_y.view(
            batch_ref_domain2each_y.shape[0]*batch_ref_domain2each_y.shape[1])

        if self.flag_erm:
            # @FIXME: check if batch_ref_domain2each_y is
            # continuous number which means it is at its initial value,
            # not yet filled
            loss_erm_match_tensor, *_ = self.model.cal_task_loss(
                batch_tensor_ref_domain2each, batch_ref_domain2each_y.long())

        # Creating tensor of shape (domain size, total domains, feat size )
        # The match tensor's first two dimension
        # [(Ref domain size) * (# train domains)]
        # has been clamped together to get features extracted
        # through self.model

        # it has to be reshaped into the match tensor shape, the same
        # for the extracted feature here, it has to reshaped into
        # the shape of the match tensor
        # to make sure that the reshape only happens at the
        # first two dimension, the feature dim has to be kept intact
        dim_feat = batch_feat_ref_domain2each.shape[1]
        num_domain_tr = len(self.task.list_domain_tr)
        batch_feat_ref_domain2each = batch_feat_ref_domain2each.view(
            curr_batch_size, num_domain_tr, dim_feat)

        batch_ref_domain2each_y = batch_ref_domain2each_y.view(
            curr_batch_size, num_domain_tr)

        # The match tensor's first two dimension
        # [(Ref domain size) * (# train domains)] has been clamped
        # together to get features extracted through self.model
        batch_tensor_ref_domain2each = \
            batch_tensor_ref_domain2each.view(curr_batch_size,
                                              num_domain_tr,
                                              batch_tensor_ref_domain2each.shape[1],   # channel
                                              batch_tensor_ref_domain2each.shape[2],   # img_h
                                              batch_tensor_ref_domain2each.shape[3])   # img_w

        # Contrastive Loss: class \times domain \times domain
        counter_same_cls_diff_domain = 1
        logger = Logger.get_logger()
        for y_c in range(self.task.dim_y):

            subset_same_cls = (batch_ref_domain2each_y[:, 0] == y_c)
            subset_diff_cls = (batch_ref_domain2each_y[:, 0] != y_c)
            feat_same_cls = batch_feat_ref_domain2each[subset_same_cls]
            feat_diff_cls = batch_feat_ref_domain2each[subset_diff_cls]
            logger.debug(f'class {y_c} with same class and different class: ' +
                         f'{feat_same_cls.shape[0]} {feat_diff_cls.shape[0]}')

            if feat_same_cls.shape[0] == 0 or feat_diff_cls.shape[0] == 0:
                logger.debug(f"no instances of label {y_c}"
                             f"in the current batch, continue")
                continue

            if torch.sum(torch.isnan(feat_diff_cls)):
                raise RuntimeError('feat_diff_cls has nan entrie(s)')

            feat_diff_cls = feat_diff_cls.view(
                feat_diff_cls.shape[0]*feat_diff_cls.shape[1],
                feat_diff_cls.shape[2])

            for d_i in range(feat_same_cls.shape[1]):
                dist_diff_cls_same_domain = dist_pairwise_cosine(
                    feat_same_cls[:, d_i, :], feat_diff_cls[:, :])

                if torch.sum(torch.isnan(dist_diff_cls_same_domain)):
                    raise RuntimeError('dist_diff_cls_same_domain NAN')

                # iterate other domains
                for d_j in range(feat_same_cls.shape[1]):
                    if d_i >= d_j:
                        continue
                    dist_same_cls_diff_domain = dist_cosine_agg(feat_same_cls[:, d_i, :],
                                                                feat_same_cls[:, d_j, :])

                    if torch.sum(torch.isnan(dist_same_cls_diff_domain)):
                        raise RuntimeError('dist_same_cls_diff_domain NAN')

                    # CTR (contrastive) loss is exclusive for
                    # CTR phase and ERM phase

                    if self.flag_erm:
                        list_batch_loss_ctr.append(torch.sum(dist_same_cls_diff_domain))
                    else:
                        i_dist_same_cls_diff_domain = 1.0 - dist_same_cls_diff_domain
                        i_dist_same_cls_diff_domain = \
                            i_dist_same_cls_diff_domain / self.aconf.tau
                        partition = torch.log(torch.exp(i_dist_same_cls_diff_domain) +
                                              dist_diff_cls_same_domain)
                        list_batch_loss_ctr.append(
                            -1 * torch.sum(i_dist_same_cls_diff_domain - partition))

                    counter_same_cls_diff_domain += dist_same_cls_diff_domain.shape[0]

        loss_ctr = sum(list_batch_loss_ctr) / counter_same_cls_diff_domain

        if self.flag_erm:
            epos = self.aconf.epos
        else:
            epos = self.aconf.epochs_ctr
        coeff = (epoch + 1)/(epos + 1)
        # loss aggregation is over different domain
        # combinations of the same batch
        # https://discuss.pytorch.org/t/leaf-variable-was-used-in-an-inplace-operation/308
        # Loosely, tensors you create directly are leaf variables.
        # Tensors that are the result of a differentiable operation are
        # not leaf variables

        if self.flag_erm:
            # extra loss of ERM phase: the ERM loss
            # (the CTR loss for the ctr phase and erm phase are different)
            # erm loss comes from two different data loaders,
            # one is rnd (random) data loader
            # the other one is the data loader from the match tensor
            loss_e = torch.tensor(0.0, requires_grad=True) + \
                    torch.mean(loss_erm_rnd_loader) + \
                    torch.mean(loss_erm_match_tensor) + \
                    self.lambda_ctr * coeff * loss_ctr
        else:
            loss_e = torch.tensor(0.0, requires_grad=True) + \
                self.lambda_ctr * coeff * loss_ctr
        # @FIXME: without torch.tensor(0.0), after a few epochs,
        # error "'float' object has no attribute 'backward'"

        loss_e.backward(retain_graph=False)
        self.optimizer.step()
        self.epo_loss_tr += loss_e.detach().item()

        torch.cuda.empty_cache()

    def mk_match_tensor(self, epoch):
        """
        initialize or update match tensor
        """
        obj_match = MatchPair(self.task.dim_y,
                              self.task.isize.i_c,
                              self.task.isize.i_h,
                              self.task.isize.i_w,
                              self.aconf.bs,
                              virtual_ref_dset_size=self.base_domain_size,
                              num_domains_tr=len(self.task.list_domain_tr),
                              list_tr_domain_size=self.list_tr_domain_size)

        # @FIXME: what is the usefulness of (epoch > 0) as argument
        self.tensor_ref_domain2each_domain_x, self.tensor_ref_domain2each_domain_y = \
        obj_match(
            self.device,
            self.task.loader_tr,
            self.model.extract_semantic_feat,
            (epoch > 0))

    def before_tr(self):
        """
        override abstract method
        """
        logger = Logger.get_logger()
        logger.info("\n\nPhase 1 start: contractive alignment without task loss: \n\n")
        # phase 1: contrastive learning
        # different than phase 2, ctr_model has no classification loss
