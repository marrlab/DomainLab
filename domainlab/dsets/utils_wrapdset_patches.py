"""
upon a task, if jigen is chosen as the algorithm, then task's dataset has to be augmented to
include tile permutation
"""
import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils import data as torchdata


class WrapDsetPatches(torchdata.Dataset):
    """
    given dataset of images, return permuations of tiles of images re-weaved
    """
    def __init__(self, dataset,
                 num_perms2classify,
                 prob_no_perm,
                 grid_len,
                 ppath=None,
                 flag_do_not_weave_tiles=False):
        """
        :param prob_no_perm: probability of no permutation: permutation will change the image, so
        the class label classifier will behave very differently compared to no permutation
        """
        if ppath is None and grid_len != 3:
            raise RuntimeError("please provide npy file of numpy array with each row \
                               being a permutation of the number of tiles, currently \
                               we only support grid length 3")
        self.dataset = dataset
        self._to_tensor = transforms.Compose([transforms.ToTensor()])
        self.arr1perm_per_row = self.__retrieve_permutations(
            num_perms2classify, ppath)
        # for 3*3 tiles, there are 9*8*7*6*5*...*1 >> 100,
        # we load from disk instead only 100 permutations
        # each row of the loaded array is a permutation of the 3*3 tile
        # of the original image
        self.grid_len = grid_len
        # break the image into 3*3 tiles
        self.prob_no_perm = prob_no_perm
        if flag_do_not_weave_tiles:
            self.fun_weave_imgs = lambda x: x
        else:
            def make_grid(img):
                """
                sew tiles together to be an image
                """
                return torchvision.utils.make_grid(
                    img, nrow=self.grid_len, padding=0)
            self.fun_weave_imgs = make_grid

    def get_tile(self, img, ind_tile):
        """
        assume a square image?
        """
        img_height = img.shape[-1]
        # @FIXME: use a better way to decide the image size
        num_tiles = float(img_height) / self.grid_len
        num_tiles = float(int(num_tiles)) + 1
        # @FIXME: extra line to ensure num_tiles=75 instead of sometimes 74
        # so torch.stack can fail in original data,
        # num_tiles = float(img.size[0]) / self.grid_len = 225/3 = 75.0
        # is an integer, but this can not be true for other cases
        ind_vertical = int(ind_tile / self.grid_len)
        ind_horizontal = ind_tile % self.grid_len
        functor_tr = transforms.ToPILImage()
        img_pil = functor_tr(img)
        # PIL.crop((left, top, right, bottom))
        # get rectangular region from box  of [left, upper, right, lower]
        tile = img_pil.crop([ind_horizontal * num_tiles,
                             ind_vertical * num_tiles,
                             (ind_horizontal + 1) * num_tiles,
                             (ind_vertical + 1) * num_tiles])
        tile = self._to_tensor(tile)
        return tile

    def __getitem__(self, index):
        img, label, *domain = self.dataset.__getitem__(index)
        if domain:
            dlabel = domain[0]
        else:
            dlabel = None
        num_grids = self.grid_len ** 2
        # divide image into grid_len^2 tiles
        list_tiles = [None] * num_grids
        # list of length num_grids of image tiles
        for ind_tile in range(num_grids):
            list_tiles[ind_tile] = self.get_tile(img, ind_tile)    # populate tile list
        ind_which_perm = np.random.randint(
            self.arr1perm_per_row.shape[0] + 1)
        # +1 in line above is for when image is not permutated, which
        # also need to be classified corrected by the permutation classifier
        # let len(self.arr1perm_per_row)=31
        # so ind_which_perm is a random number in [0, 31] in total 31+1 classes
        # ind_which_perm is basically the row index to choose
        # from self.arr1perm_per_row which is a matrix of 31*9
        # where 9=3*3 is the number of tiles the image is broken into
        if self.prob_no_perm:  # probability of no permutation of tiles
            # note that this "if" block is not redundant: permutation will change the image
            # thus change the behavior of the class label classifier, if self.prob_no_perm=1.0
            # then the algorithm will behave similarly to deepall, though not completely same
            # FIXME: what hyperparameters one could set to let jigen=deepall?
            if self.prob_no_perm > np.random.rand():
                ind_which_perm = 0
        # ind_which_perm = 0 means no permutation, the classifier need to
        # judge if the image has not been permutated as well
        list_reordered_tiles = None
        if ind_which_perm == 0:
            list_reordered_tiles = list_tiles  # no permutation of images
        else:   # default
            perm_chosen = self.arr1perm_per_row[ind_which_perm - 1]
            list_reordered_tiles = [list_tiles[perm_chosen[ind_tile]]
                                    for ind_tile in range(num_grids)]
        stacked_tiles = torch.stack(list_reordered_tiles, 0)
        # NOTE: stacked_tiles will be [9, 3, 30, 30], which will be weaved to
        # be a whole image again by self.fun_weave_imgs
        # NOTE: ind_which_perm = 0 means no permutation, the classifier need to
        # judge if the image has not been permutated as well

        return self.fun_weave_imgs(stacked_tiles), label, dlabel, int(ind_which_perm)
        # ind_which_perm is the ground truth for the permutation index

    def __len__(self):
        return self.dataset.__len__()

    def __retrieve_permutations(self, num_perms_as_classes, ppath=None):
        """
        for 9 tiles which partition the image, we have num_perms_as_classes
        number of different permutations of the tiles, the classifier will
        classify the re-tile-ordered image permutation it come from.
        """
        # @FIXME: this assumes always a relative path
        mdir = os.path.dirname(os.path.realpath(__file__))
        if ppath is None:
            ppath = f'data/patches_permutation4jigsaw/permutations_{num_perms_as_classes}.npy'
        mpath = os.path.join(mdir, "..", "..", ppath)
        arr_permutation_rows = np.load(mpath)
        # from range [1,9] to [0,8] since python array start with 0
        if arr_permutation_rows.min() == 1:
            arr_permutation_rows = arr_permutation_rows - 1
        return arr_permutation_rows
