"""
upon a task, if jigen is chosen as the algorithm, then task's dataset has to be augmented to
include tile permutation
"""
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils import data as torchdata


GTRANS4TILE = transforms.Compose([
    transforms.RandomGrayscale(0.1),
    # @FIXME: this is cheating for jiGen
    # but seems to have a big impact on performance
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class WrapDsetPatches(torchdata.Dataset):
    """
    given dataset of images, return permuations of tiles of images re-weaved
    """
    def __init__(self, dataset,
                 num_perms2classify=31,
                 transform4tile=GTRANS4TILE,
                 flag_do_not_weave_tiles=False,
                 prob_no_perm=0.7,
                 grid_len=3):
        self.dataset = dataset
        self.arr_perm_rows = self.__retrieve_permutations(
            num_perms2classify)
        # for 3*3 tiles, there are 9*8*7*6*5*...*1 >> 100,
        # we load from disk instead only 100 permutations
        self.grid_len = grid_len
        # break the image into 3*3 tiles
        self.prob_no_perm = prob_no_perm
        self._transform4tile = transform4tile
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
        tile = self._transform4tile(tile)
        return tile

    def __getitem__(self, index):
        img, label, *_ = self.dataset.__getitem__(index)
        num_grids = self.grid_len ** 2    # divide image into grid_len^2 tiles
        list_tiles = [None] * num_grids     # list of length num_grids of image tiles
        for ind_tile in range(num_grids):
            list_tiles[ind_tile] = self.get_tile(img, ind_tile)    # populate tile list
        ind_which_perm = np.random.randint(
            self.arr_perm_rows.shape[0] + 1)  # added 1 for class 0: unsorted
        # len(self.arr_perm_rows) by default is 100,
        # so ind_which_perm is a random number between 0 and 101
        # ind_which_perm is basically the row index to choose
        # from self.arr_perm_rows which is a matrix of 100*9 usually,
        # where 9=3*3 is
        # the number of tiles the image is broken into
        if self.prob_no_perm:    # default is None
            if self.prob_no_perm > np.random.rand():
                ind_which_perm = 0
        list_reordered_tiles = None
        if ind_which_perm == 0:
            list_reordered_tiles = list_tiles  # no permutation
        else:   # default
            perm_chosen = self.arr_perm_rows[ind_which_perm - 1]
            list_reordered_tiles = [list_tiles[perm_chosen[ind_tile]]
                                    for ind_tile in range(num_grids)]

        stacked_tiles = torch.stack(list_reordered_tiles, 0)
        # the 0th dim is the batch dimension
        # NOTE: label must be the second place so that functions like
        # performance.get_accuracy could work!
        return self.fun_weave_imgs(stacked_tiles), label, int(ind_which_perm)
        # ind_which_perm is the ground truth for the permutation index

    def __len__(self):
        return self.dataset.__len__()

    def __retrieve_permutations(self, num_perms_as_classes):
        """
        for 9 tiles which partition the image, we have num_perms_as_classes
        number of different permutations of the tiles, the classifier will
        classify the re-tile-ordered image permutation it come from.
        """
        # @FIXME: this assumes always a relative path
        mpath = f'data/patches_permutation4jigsaw/permutations_{num_perms_as_classes}.npy'
        arr_permutation_rows = np.load(mpath)
        # from range [1,9] to [0,8]
        if arr_permutation_rows.min() == 1:
            arr_permutation_rows = arr_permutation_rows - 1
        return arr_permutation_rows
