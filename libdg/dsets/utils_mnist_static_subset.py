import os
import numpy as np


class MnistStaticSubsetNumpy():
    """
    Original MNIST dataset has 60K images which can be too much.
    This class use pre-sampled indexes to subset the original MNIST dataset.
    source: https://github.com/AMLab-Amsterdam/DIVA/tree/master/paper_experiments/rotated_mnist/dataset
    with commit hash tag [ab590b4c95b5f667e7b5a7730a797356d124].
    The subset indexes are provided as pre-stored npy files
    In total there are 10 subset indexes corresponding to 10 random seeds
    """

    def __init__(self, dir_npy, seed):
        """
        :param seed: from 0 to 9.
        :param dir_npy:  The subset indexes are provided as pre-stored npy files
        """
        self.dir_npy = os.path.expanduser(dir_npy)
        self.seed = seed

    def __call__(self):
        """
        return numpy array of indexes
        """
        fname = 'supervised_inds_' + str(self.seed) + '.npy'
        fpath = os.path.join(self.dir_npy, fname)
        return np.load(fpath)
