"""
create dictionary for matching
"""
import torch
from domainlab.utils.logger import Logger


class MatchDictInit():
    """
    base class for matching dictionary creator
    """
    def __init__(self, keys, vals, i_c, i_h, i_w):
        self.keys = keys
        self.vals = vals
        self.i_c = i_c
        self.i_h = i_h
        self.i_w = i_w

    def get_num_rows(self, key):
        raise NotImplementedError

    def __call__(self):
        dict_data = {}
        for key in self.keys:
            dict_data[key] = {}
            num_rows = self.get_num_rows(key)
            dict_data[key]['data'] = torch.rand((num_rows, self.i_c, self.i_w, self.i_h))
            # @FIXME: some labels won't be filled at all, when using training loader since the incomplete batch is dropped
            dict_data[key]['label'] = torch.rand((num_rows, 1))  # scalar label
            dict_data[key]['idx'] = torch.randint(low=0, high=1, size=(num_rows, 1))
        return dict_data


class  MatchDictVirtualRefDset2EachDomain(MatchDictInit):
    """
    dict[0:virtual_ref_dset_size] has tensor dimension: (num_domains_tr, i_c, i_h, i_w)
    """
    def __init__(self, virtual_ref_dset_size, num_domains_tr, i_c, i_h, i_w):
        """
        virtual_ref_dset_size is a virtual dataset, len(virtual_ref_dset_size) = sum of all popular domains
        """
        super().__init__(keys=range(virtual_ref_dset_size), vals=num_domains_tr,
                         i_c=i_c, i_h=i_h, i_w=i_w)

    def get_num_rows(self, key=None):
        """
        key is 0:virtual_ref_dset_size
        """
        return self.vals   # total_domains


class MatchDictNumDomain2SizeDomain(MatchDictInit):
    """
    tensor dimension for the kth domain: [num_domains_tr, (size_domain_k, i_c, i_h, i_w)]
    """
    def __init__(self, num_domains_tr, list_tr_domain_size, i_c, i_h, i_w):
        super().__init__(keys=range(num_domains_tr), vals=list_tr_domain_size,
                         i_c=i_c, i_h=i_h, i_w=i_w)

    def get_num_rows(self, key):
        return self.vals[key]   # list_tr_domain_size[domain_index]


def dist_cosine_agg(x1, x2):
    """
    torch.nn.CosineSimilarity assumes x1 and x2 share exactly the same dimension
    """
    fun_cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
    return 1.0 - fun_cos(x1, x2)

def fun_tensor_normalize(tensor_batch_x):
    eps = 1e-8
    batch_norm_x = tensor_batch_x.norm(dim=1)   # Frobenius norm or Euclidean Norm long the embedding direction, len(norm) should be batch_size
    batch_norm_x = batch_norm_x.view(batch_norm_x.shape[0], 1)  # add dimension to tensor
    tensor_eps = eps * torch.ones_like(batch_norm_x)
    tensor_batch_x = tensor_batch_x / torch.max(batch_norm_x, tensor_eps)
    assert not torch.sum(torch.isnan(tensor_batch_x))
    return tensor_batch_x


def dist_pairwise_cosine(x1, x2, tau=0.05):
    """
    x1 and x2 does not necesarilly have the same shape, and we want to have a cartesian product of the pairwise distances
    """
    assert len(x1.shape) == 2 and len(x2.shape) == 2
    assert not torch.sum(torch.isnan(x1))
    assert not torch.sum(torch.isnan(x2))

    x1 = fun_tensor_normalize(x1)
    x2 = fun_tensor_normalize(x2)

    x1_extended_dim = x1.unsqueeze(1)  # Returns a new tensor with a dimension of size one inserted at the specified position.
    # extend the order of by insering a new dimension so that cartesion product of pairwise distance can be calculated

    # since the batch size of x1 and x2 won't be the same, directly calculting elementwise product will cause an error
    # with order 3 multiply order 2 tensor, the feature dimension will be matched then the rest dimensions form cartesian product
    cos_sim = torch.sum(x1_extended_dim*x2, dim=2)   # elementwise product
    cos_sim = cos_sim / tau  # make cosine similarity bigger than 1
    assert not torch.sum(torch.isnan(cos_sim))
    loss = torch.sum(torch.exp(cos_sim), dim=1)
    assert not torch.sum(torch.isnan(loss))
    return loss


def get_base_domain_size4match_dg(task):
    """
    Base domain is a dataset where each class
    set come from one of the nominal domains
    """
    # @FIXME: base domain should be calculated only on training domains
    # instead of all the domains!
    # domain_keys = task.get_list_domains()
    domain_keys = task.list_domain_tr
    base_domain_size = 0
    classes = task.list_str_y
    for mclass in classes:
        num = 0
        ref_domain = -1
        for _, domain_key in enumerate(domain_keys):
            if task.dict_domain_class_count[domain_key][mclass] > num:
                ref_domain = domain_key
                num = task.dict_domain_class_count[domain_key][mclass]
        logger = Logger.get_logger()
        logger.info(f"for class {mclass} bigest sample size is {num} "
                    f"ref domain is {ref_domain}")
        base_domain_size += num
    return base_domain_size