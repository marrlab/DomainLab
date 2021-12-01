from torch.utils.data import Dataset


class DsetIndDecorator4XYD(Dataset):
    """
    For dataset of x, y, d,  decorate it wih index
    """
    def __init__(self, dset):
        """
        :param dset: x,y,d
        """
        tuple_m = dset[0]
        if len(tuple_m) != 3:
            raise RuntimeError(
                "dataset to be wrapped should output x,y,d, got length ", len(tuple_m))
        self.dset = dset

    def __getitem__(self, index):
        """
        :param index:
        """
        tensor_x, vec_y, vec_d = self.dset.__getitem__(index)
        return tensor_x, vec_y, vec_d, index

    def __len__(self):
        return self.dset.__len__()
