import torch
from torch.utils.data import Dataset
from os.path import join
from os import listdir
import numpy as np

class SoundDataset(Dataset):
    def __init__(self,
                 source_dir
                ):
        super().__init__()
        # TODO - installation script on nonexistent or empty source
        self.source_dir = source_dir
        self.filenames = listdir(source_dir) # Note this is deterministic, but unintuitive (sort todo?)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        full_path = join(self.source_dir, self.filenames[index])
        return torch.tensor(np.load(full_path))