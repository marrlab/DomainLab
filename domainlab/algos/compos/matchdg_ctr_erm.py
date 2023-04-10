import torch

from domainlab.algos.compos.matchdg_base import MatchAlgoBase
from domainlab.algos.compos.matchdg_utils import (dist_cosine_agg,
                                                  dist_pairwise_cosine)
from domainlab.algos.trainers.a_trainer import mk_opt


class MatchCtrErm(MatchAlgoBase):
    """
    Contrastive Learning
    """
    def __init__(self, task, phi, args, device, exp, flag_erm=False, opt=None):
        """
        """
        if opt is None:
            opt = mk_opt(phi, args)
        super().__init__(task, phi, args, device, exp, opt)
        self.epo_loss_tr = 0
        self.flag_erm = flag_erm
        self.epos = self.args.epochs_ctr
        self.epos_per_match = self.args.epos_per_match_update
        self.str_phase = "ctr"
        self.lambda_ctr = 1.0
        if self.flag_erm:
            self.lambda_ctr = self.args.gamma_reg
            self.epos = self.args.epos - self.args.epochs_ctr
            self.str_phase = "erm"
            self.init_erm_phase()
        else:
            self.mk_match_tensor(epoch=0)
        self.flag_stop = False
        self.tuple_tensor_ref_domain2each_y = None
        self.tuple_tensor_refdomain2each = None

    def train(self):
        """
        train self.epos number of epochs
        """
        for epoch in range(self.epos):
            self.tr_epoch(epoch)

    def tr_epoch(self, epoch):
        """
        # data in one batch comes from two sources: one part from loader,
        # the other part from match tensor
        """
        self.epo_loss_tr = 0
        print(self.str_phase, "epoch", epoch)
        # update match tensor
        if (epoch + 1) % self.epos_per_match == 0:
            self.mk_match_tensor(epoch)

        inds_shuffle = torch.randperm(self.tensor_ref_domain2each_domain_x.size(0))
        # NOTE: match tensor size: N(ref domain size) * #(train domains) * (image size: c*h*w)
        # self.tensor_ref_domain2each_domain_x[inds_shuffle]
        # shuffles the match tensor at the first dimension
        self.tuple_tensor_refdomain2each = torch.split(
            self.tensor_ref_domain2each_domain_x[inds_shuffle],
            self.args.bs, dim=0)
        # Splits the tensor into chunks.
        # Each chunk is a view of the original tensor of batch size self.args.bs
        # return is a tuple of the splited chunks
        self.tuple_tensor_ref_domain2each_y = torch.split(
            self.tensor_ref_domain2each_domain_y[inds_shuffle],
            self.args.bs, dim=0)
        print("number of batches in match tensor: ", len(self.tuple_tensor_refdomain2each))
        print("single batch match tensor size: ", self.tuple_tensor_refdomain2each[0].shape)

        for batch_idx, (x_e, y_e, d_e, *_) in enumerate(self.loader):
            # random loader with same batch size as the match tensor loader
            # the 4th output of self.loader is not used at all,
            # is only used for creating the match tensor
            self.update_batch(epoch, batch_idx, x_e, y_e, d_e)
            if self.flag_stop is True:
                print("ref/base domain vs each domain match \
                      traversed one sweep, starting new epoch")
                break
        if not self.flag_erm:
            # Save ctr model's weights post each epoch
            self.save_model_ctr_phase()

    def update_batch(self, epoch, batch_idx, x_e, y_e, d_e):
        """
        update network for each batch
        """
        self.opt.zero_grad()
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
            loss_erm_rnd_loader = self.phi.cal_loss(x_e, y_e, d_e)

        num_batches = len(self.tuple_tensor_refdomain2each)

        if batch_idx >= num_batches:
            print("ref/base domain vs each domain match \
                    traversed one sweep, starting new epoch")
            self.flag_stop = True
            return

        curr_batch_size = self.tuple_tensor_refdomain2each[batch_idx].shape[0]

        batch_tensor_ref_domain2each = self.tuple_tensor_refdomain2each[batch_idx].to(self.device)
        # make order 5 tensor: (ref_domain, domain, channel, img_h, img_w)
        # with first dimension as batch size

        # clamp the first two dimensions so the phi network could map image to feature
        batch_tensor_ref_domain2each = batch_tensor_ref_domain2each.view(
            batch_tensor_ref_domain2each.shape[0]*batch_tensor_ref_domain2each.shape[1],
            batch_tensor_ref_domain2each.shape[2],   # channel
            batch_tensor_ref_domain2each.shape[3],   # img_h
            batch_tensor_ref_domain2each.shape[4])   # img_w
        # now batch_tensor_ref_domain2each first dim will not be batch_size!
        # batch_tensor_ref_domain2each.shape torch.Size([40, channel, 224, 224])

        batch_feat_ref_domain2each = self.phi.extract_semantic_feat(
            batch_tensor_ref_domain2each)
        # batch_feat_ref_domain2each.shape torch.Size[40, 512]
        # torch.sum(torch.isnan(batch_tensor_ref_domain2each))
        # assert not torch.sum(torch.isnan(batch_feat_ref_domain2each))
        flag_isnan = torch.any(torch.isnan(batch_feat_ref_domain2each))
        if flag_isnan:
            print(batch_tensor_ref_domain2each)
            raise RuntimeError("batch_feat_ref_domain2each NAN! is learning rate too big or \
                               hyper-parameter tau not set appropriately?")

        # for contrastive training phase,
        # the last layer of the model is replaced with identity

        batch_ref_domain2each_y = self.tuple_tensor_ref_domain2each_y[batch_idx].to(self.device)
        batch_ref_domain2each_y = batch_ref_domain2each_y.view(
            batch_ref_domain2each_y.shape[0]*batch_ref_domain2each_y.shape[1])

        if self.flag_erm:
            # @FIXME: check if batch_ref_domain2each_y is
            # continuous number which means it is at its initial value,
            # not yet filled
            loss_erm_match_tensor = self.phi.cal_loss(
                batch_tensor_ref_domain2each, batch_ref_domain2each_y.long())

        # Creating tensor of shape (domain size, total domains, feat size )
        # The match tensor's first two dimension
        # [(Ref domain size) * (# train domains)]
        # has been clamped together to get features extracted
        # through self.phi

        # it has to be reshaped into the match tensor shape, the same
        # for the extracted feature here, it has to reshaped into
        # the shape of the match tensor
        # to make sure that the reshape only happens at the
        # first two dimension, the feature dim has to be kept intact
        dim_feat = batch_feat_ref_domain2each.shape[1]
        batch_feat_ref_domain2each = batch_feat_ref_domain2each.view(
            curr_batch_size, self.num_domain_tr, dim_feat)

        batch_ref_domain2each_y = batch_ref_domain2each_y.view(
            curr_batch_size, self.num_domain_tr)

        # The match tensor's first two dimension
        # [(Ref domain size) * (# train domains)] has been clamped
        # together to get features extracted through self.phi
        batch_tensor_ref_domain2each = \
            batch_tensor_ref_domain2each.view(curr_batch_size,
                                              self.num_domain_tr,
                                              batch_tensor_ref_domain2each.shape[1],   # channel
                                              batch_tensor_ref_domain2each.shape[2],   # img_h
                                              batch_tensor_ref_domain2each.shape[3])   # img_w

        # Contrastive Loss: class \times domain \times domain
        counter_same_cls_diff_domain = 1
        for y_c in range(self.dim_y):

            subset_same_cls = (batch_ref_domain2each_y[:, 0] == y_c)
            subset_diff_cls = (batch_ref_domain2each_y[:, 0] != y_c)
            feat_same_cls = batch_feat_ref_domain2each[subset_same_cls]
            feat_diff_cls = batch_feat_ref_domain2each[subset_diff_cls]
            # print('class', y_c, "with same class and different class: ",
            #      feat_same_cls.shape[0], feat_diff_cls.shape[0])

            if feat_same_cls.shape[0] == 0 or feat_diff_cls.shape[0] == 0:
                # print("no instances of label",
                # y_c,
                # "in the current batch, continue")
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
                            i_dist_same_cls_diff_domain / self.args.tau
                        partition = torch.log(torch.exp(i_dist_same_cls_diff_domain) +
                                              dist_diff_cls_same_domain)
                        list_batch_loss_ctr.append(
                            -1 * torch.sum(i_dist_same_cls_diff_domain - partition))

                    counter_same_cls_diff_domain += dist_same_cls_diff_domain.shape[0]

        loss_ctr = sum(list_batch_loss_ctr) / counter_same_cls_diff_domain

        coeff = (epoch + 1)/(self.epos + 1)
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
        self.opt.step()
        self.epo_loss_tr += loss_e.detach().item()

        torch.cuda.empty_cache()
