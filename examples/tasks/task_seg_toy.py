import torch
from torch.utils.data import Dataset
import numpy as np

from domainlab.tasks.task_dset import mk_task_dset
from domainlab.tasks.utils_task import ImSize


class ToySegmentationDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=(64, 64), noise=0.1):
        self.num_samples = num_samples
        self.image_size = image_size
        self.noise = noise
        self.generate_data()

    def generate_data(self):
        # Generate random images
        self.images = torch.rand(self.num_samples, 1, *self.image_size) # FIXME: we have to change this to 3 channels
        # Generate random segmentation masks
        self.masks = torch.zeros(self.num_samples, 1, *self.image_size)
        for i in range(self.num_samples):
            # Randomly select a segment size
            segment_size = np.random.randint(10, 20)
            # Randomly select a position for the segment
            x = np.random.randint(0, self.image_size[0] - segment_size)
            y = np.random.randint(0, self.image_size[1] - segment_size)
            # Add the segment to the mask
            self.masks[i, 0, x:x+segment_size, y:y+segment_size] = 1
        # Add noise to the masks
        self.masks += self.noise * torch.randn_like(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


task = mk_task_dset(isize=ImSize(1, 64, 64), dim_y=2, taskna="custom_task") # FIXME: remove dim_y
task.add_domain(
    name="domain1",
    dset_tr=ToySegmentationDataset(10),
    dset_val=ToySegmentationDataset(5),
)

task.add_domain(
    name="domain2",
    dset_tr=ToySegmentationDataset(15),
    dset_val=ToySegmentationDataset(8),
)

task.add_domain(
    name="domain3",
    dset_tr=ToySegmentationDataset(20),
    dset_val=ToySegmentationDataset(10),
)

def get_task(na=None):
    return task
