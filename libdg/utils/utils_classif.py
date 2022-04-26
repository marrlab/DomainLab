import numpy as np
import torch
from torch.nn import functional as F


def mk_dummy_label_list_str(prefix, dim):
    """
    only used for testing, to generate list of class/domain label names
    """
    return [prefix + str(i) for i in range(dim)]


def logit2preds_vpic(logit):
    """
    :logit: batch of logit vector
    :return: vector of one-hot,
             vector of probability,
             index,
             maximum probability
    """
    mat_prob = F.softmax(logit, dim=1)
    # get the index of the maximum softmax probability
    max_prob, max_ind = torch.topk(mat_prob, 1)
    # convert the digit(s) to one-hot tensor(s)
    one_hot = logit.new_zeros(mat_prob.size())
    one_hot = one_hot.scatter_(dim=1, index=max_ind, value=1.0)
    return one_hot, mat_prob, max_ind, max_prob

def get_label_na(tensor_ind, list_str_na):
    """
    given list of label names in strings, map tensor of index to label names
    """
    arr_ind_np = tensor_ind.cpu().numpy()
    arr_ind = np.squeeze(arr_ind_np, axis=1)   # explicitly use axis=1 to deal with edge case of only
    # instance left
    # list_ind = list(arr_ind): if there is only dimension 1 tensor_ind, then there is a problem
    list_ind = arr_ind.tolist()
    list_na = [list_str_na[ind] for ind in list_ind]
    return list_na
