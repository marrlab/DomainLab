import warnings
import numpy as np
import torch
from libdg.algos.compos.matchdg_utils import MatchDictVirtualRefDset2EachDomain
from libdg.algos.compos.matchdg_utils import MatchDictNumDomain2SizeDomain
from libdg.utils.utils_class import store_args


class MatchPair():
    @store_args
    def __init__(self, dim_y, i_c, i_h, i_w, bs_match, virtual_ref_dset_size, num_domains_tr,
                 list_tr_domain_size):
        """
        :param virtual_ref_dset_size:  sum of biggest class sizes
        :param num_domains_tr:
        :param list_tr_domain_size:
        :param phi: neural network to generate causal features from input
        """
        self.dict_cls_ind_base_domain_ind = {}
        self.dict_virtual_dset2each_domain = MatchDictVirtualRefDset2EachDomain(
            virtual_ref_dset_size=virtual_ref_dset_size,
            num_domains_tr=num_domains_tr,
            i_c=i_c, i_h=i_h, i_w=i_w)()

        self.dict_domain_data = MatchDictNumDomain2SizeDomain(
            num_domains_tr=num_domains_tr,
            list_tr_domain_size=list_tr_domain_size,
            i_c=i_c, i_h=i_h, i_w=i_w)()

        self.indices_matched = {}
        for key in range(virtual_ref_dset_size):
            self.indices_matched[key] = []
        self.perfect_match_rank = []

        self.domain_count = {}
        for domain in range(num_domains_tr):
            self.domain_count[domain] = 0

    def _fill_data(self, loader):
        """
        copy all data from loader, then store them in memory variable self.dict_domain_data
        """
        for _, (x_e, y_e, d_e, idx_e) in enumerate(loader):
            # traverse mixed domain data from loader
            x_e = x_e
            y_e = torch.argmax(y_e, dim=1)
            d_e = torch.argmax(d_e, dim=1).numpy()
            unique_domains = np.unique(d_e)   # get all domains in current batch
            for domain_idx in unique_domains:
                flag_curr_domain = (d_e == domain_idx)
                # flag_curr_domain is subset indicator of True of False for selection of data from the mini-batch
                global_indices = idx_e[flag_curr_domain]
                # global_indices are subset of idx_e, which contains global index of data from the loader
                for local_ind in range(global_indices.shape[0]):
                    # FIXME: the following is just coping all data to self.dict_domain_data (in memory with ordering), which seems redundant
                    global_ind = global_indices[local_ind].item()  # tensor.item get the scalar
                    self.dict_domain_data[domain_idx]['data'][global_ind] = x_e[flag_curr_domain][local_ind]
                    # flag_curr_domain are subset indicator for selection of domain
                    self.dict_domain_data[domain_idx]['label'][global_ind] = y_e[flag_curr_domain][local_ind]
                    # copy trainining batch to dict_domain_data
                    self.dict_domain_data[domain_idx]['idx'][global_ind] = idx_e[flag_curr_domain][local_ind]
                    self.domain_count[domain_idx] += 1

        for domain in range(self.num_domains_tr):
            if self.domain_count[domain] != self.list_tr_domain_size[domain]:
                warnings.warn("domain_count show matching dictionary missing data!")

    def _cal_base_domain(self):
        """
        # Determine the base_domain_idx as the domain with the max samples of the current class
        # Create dictionary: class label -> list of ordered flag_curr_domain
        """
        for y_c in range(self.dim_y):
            base_domain_size = 0
            base_domain_idx = -1
            for domain_idx in range(self.num_domains_tr):
                flag_curr_class = (self.dict_domain_data[domain_idx]['label'] == y_c)   # tensor of True/False
                curr_size = self.dict_domain_data[domain_idx]['label'][flag_curr_class].shape[0]    # flag_curr_class are subset indicator
                if base_domain_size < curr_size:
                    base_domain_size = curr_size
                    base_domain_idx = domain_idx
            self.dict_cls_ind_base_domain_ind[y_c] = base_domain_idx
            # for each class label, there is a base domain
            print("for class", y_c)
            print("domain index as base domain:", base_domain_idx)
            print("Base Domain size", base_domain_size)



    def __call__(self, device, loader, phi, flag_match_min_dist):
        self._fill_data(loader)
        self._cal_base_domain()
        for curr_domain_ind in range(self.num_domains_tr):
            counter_ref_dset_size = 0
            for y_c in range(self.dim_y):
                base_domain_idx = self.dict_cls_ind_base_domain_ind[y_c]  # which domain to use as the base domain for the current class
                flags_base_domain_curr_cls = (self.dict_domain_data[base_domain_idx]['label'] == y_c)     # subset indicator
                flags_base_domain_curr_cls = flags_base_domain_curr_cls[:, 0]
                global_inds_base_domain_curr_cls = self.dict_domain_data[base_domain_idx]['idx'][flags_base_domain_curr_cls]
                # pick out base domain class label y_c images
                # the difference of this block is "curr_domain_ind" in iteration is used instead of base_domain_idx for current class
                flag_curr_domain_curr_cls = (self.dict_domain_data[curr_domain_ind]['label'] == y_c)     # pick out current domain y_c class images
                # NO label matches y_c
                flag_curr_domain_curr_cls = flag_curr_domain_curr_cls[:, 0]
                global_inds_curr_domain_curr_cls = self.dict_domain_data[curr_domain_ind]['idx'][flag_curr_domain_curr_cls]
                size_curr_domain_curr_cls = global_inds_curr_domain_curr_cls.shape[0]
                if size_curr_domain_curr_cls == 0:  # there is no class y_c in current domain
                    print("current domain", curr_domain_ind, " does not contain class ", y_c)
                    continue

                # compute base domain features for class label y_c
                x_base_domain_curr_cls = self.dict_domain_data[base_domain_idx]['data'][flags_base_domain_curr_cls]
                # pick out base domain class label y_c images
                # split data into chunks
                tuple_batch_x_base_domain_curr_cls = torch.split(x_base_domain_curr_cls, self.bs_match, dim=0)
                list_base_feat = []
                for batch_x_base_domain_curr_cls in tuple_batch_x_base_domain_curr_cls:
                    with torch.no_grad():
                        batch_x_base_domain_curr_cls = batch_x_base_domain_curr_cls.to(device)
                        feat = phi(batch_x_base_domain_curr_cls)
                        list_base_feat.append(feat.cpu())
                tensor_feat_base_domain_curr_cls = torch.cat(list_base_feat)   # base domain features

                if flag_match_min_dist:  # if epoch > 0:flag_match_min_dist=True
                    x_curr_domain_curr_cls = self.dict_domain_data[curr_domain_ind]['data'][flag_curr_domain_curr_cls]
                    # indices_curr pick out current domain y_c class images
                    tuple_x_batch_curr_domain_curr_cls = torch.split(x_curr_domain_curr_cls, self.bs_match, dim=0)
                    list_feat_x_curr_domain_curr_cls = []
                    for batch_feat in tuple_x_batch_curr_domain_curr_cls:
                        with torch.no_grad():
                            batch_feat = batch_feat.to(device)
                            out = phi(batch_feat)
                            list_feat_x_curr_domain_curr_cls.append(out.cpu())
                    tensor_feat_curr_domain_curr_cls = torch.cat(list_feat_x_curr_domain_curr_cls)
                    # feature through inference network for the current domain of class y_c

                tensor_feat_base_domain_curr_cls = tensor_feat_base_domain_curr_cls.unsqueeze(1)
                tuple_feat_base_domain_curr_cls = torch.split(tensor_feat_base_domain_curr_cls, self.bs_match, dim=0)

                counter_curr_cls_base_domain = 0
                for feat_base_domain_curr_cls in tuple_feat_base_domain_curr_cls:  # tuple_feat_base_domain_curr_cls is a tuple of splitted part

                    if flag_match_min_dist:   # if epoch > 0:flag_match_min_dist=True
                        # Need to compute over batches of feature due to device Memory out errors
                        # Else no need for loop over tuple_feat_base_domain_curr_cls;
                        # could have simply computed tensor_feat_curr_domain_curr_cls - tensor_feat_base_domain_curr_cls
                        dist_same_class_base_domain_curr_domain = torch.sum((tensor_feat_curr_domain_curr_cls - feat_base_domain_curr_cls)**2, dim=2)
                        # tensor_feat_curr_domain_curr_cls.shape torch.Size([184, 512])
                        # feat_base_domain_curr_cls.shape torch.Size([64, 1, 512])
                        # (tensor_feat_curr_domain_curr_cls - feat_base_domain_curr_cls).shape: torch.Size([64, 184, 512])
                        # dist_same_class_base_domain_curr_domain.shape: torch.Size([64, 184]) is the per element distance of the cartesian product of feat_base_domain_curr_cls vs tensor_feat_curr_domain_curr_cls
                        match_ind_base_domain_curr_domain = torch.argmin(dist_same_class_base_domain_curr_domain, dim=1)  # the batch index of the neareast neighbors
                        # len(match_ind_base_domain_curr_domain)=64
                        # theoretically match_ind_base_domain_curr_domain can be a permutation of 0 to 183 though of size 64
                        # sort_val, sort_idx = torch.sort(dist_same_class_base_domain_curr_domain, dim=1)
                        del dist_same_class_base_domain_curr_domain

                    for idx in range(feat_base_domain_curr_cls.shape[0]):  #  feat_base_domain_curr_cls.shape torch.Size([64, 1, 512])
                        # counter_curr_cls_base_domain =0 at initialization
                        global_pos_base_domain_curr_cls = global_inds_base_domain_curr_cls[counter_curr_cls_base_domain].item()   # # global_inds_base_domain_curr_cls pick out base domain class label y_c images
                        if curr_domain_ind == base_domain_idx:
                            ind_match_global_curr_domain_curr_cls = global_pos_base_domain_curr_cls
                        else:
                            if flag_match_min_dist:  # if epoch > 0:match_min_dist=True
                                ind_match_global_curr_domain_curr_cls = global_inds_curr_domain_curr_cls[match_ind_base_domain_curr_domain[idx]].item()
                            else:  # if epoch == 0
                                ind_match_global_curr_domain_curr_cls = global_inds_curr_domain_curr_cls[counter_curr_cls_base_domain%size_curr_domain_curr_cls].item()

                        self.dict_virtual_dset2each_domain[counter_ref_dset_size]['data'][curr_domain_ind] = self.dict_domain_data[curr_domain_ind]['data'][ind_match_global_curr_domain_curr_cls]
                        self.dict_virtual_dset2each_domain[counter_ref_dset_size]['label'][curr_domain_ind] = self.dict_domain_data[curr_domain_ind]['label'][ind_match_global_curr_domain_curr_cls]
                        counter_curr_cls_base_domain += 1
                        counter_ref_dset_size += 1

            if counter_ref_dset_size != self.virtual_ref_dset_size:
                warnings.warn("counter_ref_dset_size not equal to self.virtual_ref_dset_size")
                print("counter_ref_dset_size", counter_ref_dset_size)
                print("self.virtual_ref_dset_size", self.virtual_ref_dset_size)


        for key in self.dict_virtual_dset2each_domain.keys():
            if self.dict_virtual_dset2each_domain[key]['label'].shape[0] != self.num_domains_tr:
                raise RuntimeError("self.dict_virtual_dset2each_domain, one key correspond to value tensor not equal to number of training domains")

        # Sanity Check: Ensure paired points have the same class label
        wrong_case = 0
        for key in self.dict_virtual_dset2each_domain.keys():
            for d_i in range(self.dict_virtual_dset2each_domain[key]['label'].shape[0]):
                for d_j in range(self.dict_virtual_dset2each_domain[key]['label'].shape[0]):
                    if d_j > d_i:
                        if self.dict_virtual_dset2each_domain[key]['label'][d_i] != self.dict_virtual_dset2each_domain[key]['label'][d_j]:
                            wrong_case += 1
        print('Total Label MisMatch across pairs: ', wrong_case)

        list_ref_domain_each_domain = []
        list_ref_domain_each_domain_label = []
        for ind_ref_domain_key in self.dict_virtual_dset2each_domain.keys():
            list_ref_domain_each_domain.append(self.dict_virtual_dset2each_domain[ind_ref_domain_key]['data'])
            list_ref_domain_each_domain_label.append(self.dict_virtual_dset2each_domain[ind_ref_domain_key]['label'])

        tensor_ref_domain_each_domain_x = torch.stack(list_ref_domain_each_domain)
        tensor_ref_domain_each_domain_label = torch.stack(list_ref_domain_each_domain_label)

        print(tensor_ref_domain_each_domain_x.shape, tensor_ref_domain_each_domain_label.shape)

        del self.dict_domain_data
        del self.dict_virtual_dset2each_domain
        return tensor_ref_domain_each_domain_x, tensor_ref_domain_each_domain_label
