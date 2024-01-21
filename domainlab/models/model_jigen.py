"""
Jigen Model Similar to DANN model
"""
import warnings

from torch.nn import functional as F

from domainlab import g_str_cross_entropy_agg
from domainlab.dsets.utils_wrapdset_patches import WrapDsetPatches
from domainlab.models.a_model_classif import AModelClassif
from domainlab.models.model_dann import mk_dann


def mk_jigen(parent_class=AModelClassif, **kwargs):
    """
    Instantiate a JiGen model

    Details:
        The model is trained to solve two tasks:
        1. Standard image classification;
        2. Source images are decomposed into grids of patches, which are then permuted. The task
        is recovering the original image by predicting the right permutation of the patches;

        The (permuted) input data is first fed into a encoder neural network
        and then into the two classification networks.
        For more details, see:
        Carlucci, Fabio M., et al. "Domain generalization by solving jigsaw puzzles."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

    Args:
        parent_class (AModel, optional): Class object determining the task
        type. Defaults to AModelClassif.

    Returns:
        ModelJiGen: model inheriting from parent class

    Input Parameters:
        list_str_y: list of labels,
        list_str_d: list of domains,
        net_encoder: neural network (input: training data, standard and shuffled),
        net_classifier_class: neural network (input: output of net_encoder;
        output: label prediction),
        net_classifier_permutation: neural network (input: output of net_encoder;
        output: prediction of permutation index),
        coeff_reg: total_loss = img_class_loss + coeff_reg * perm_task_loss
        nperm: number of permutations to use, by default 31
        prob_permutation: probability of shuffling image tiles

    Usage:
        For a concrete example, see:
        https://github.com/marrlab/DomainLab/blob/master/tests/test_mk_exp_jigen.py
    """

    class_dann = mk_dann(parent_class, **kwargs)

    class ModelJiGen(class_dann):
        """
        Jigen Model Similar to DANN model
        """

        def __init__(
            self,
            net_encoder,
            net_classifier_permutation,
            coeff_reg,
            n_perm=31,
            prob_permutation=0.1):
            super().__init__(
                list_d_tr=None,
                alpha=coeff_reg,
                net_encoder=net_encoder,
                net_discriminator=net_classifier_permutation,
            )
            self.net_encoder = net_encoder
            self.net_classifier_permutation = net_classifier_permutation
            self.n_perm = n_perm
            self.prob_perm = prob_permutation

        def dset_decoration_args_algo(self, args, ddset):
            """
            JiGen need to shuffle the tiles of the original image
            """
            # note if model is initialized via API, args might not agree with  self.n_perm
            # in this case, we just ignore args, since in model builder, self.n_perm is set
            # via args as well
            nperm = self.n_perm  # ignore args since self.n_perm also set via args
            pperm = self.prob_perm
            ddset_new = WrapDsetPatches(
                ddset,
                num_perms2classify=nperm,
                prob_no_perm=1 - pperm,
                grid_len=args.grid_len,
                ppath=args.jigen_ppath,
            )
            return ddset_new

        def _cal_reg_loss(self, tensor_x, tensor_y, tensor_d, others):
            """
            JiGen don't need domain label but a pre-defined permutation index
            to calculate regularization loss

            We don't know if tensor_x is an original/un-tile-shuffled image or
            a permutated image, which is the task of cal_reg_loss to classify
            which permutation has been used or no permutation has been used at
            all (which also has to be classified)
            """
            _ = tensor_y
            _ = tensor_d
            if isinstance(others, list) or isinstance(others, tuple):
                vec_perm_ind = others[0]
            else:
                vec_perm_ind = others
            # tensor_x can be either original image or tile-shuffled image
            feat = self.extract_semantic_feat(tensor_x)
            logits_which_permutation = self.net_classifier_permutation(feat)
            # _, batch_target_scalar = vec_perm_ind.max(dim=1)
            batch_target_scalar = vec_perm_ind
            batch_target_scalar = batch_target_scalar.to(tensor_x.device)
            loss_perm = F.cross_entropy(
                logits_which_permutation,
                batch_target_scalar,
                reduction=g_str_cross_entropy_agg,
            )
            return [loss_perm], [self.alpha]

    return ModelJiGen
