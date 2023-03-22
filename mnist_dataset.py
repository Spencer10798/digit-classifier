import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data.iloc[idx, 0]
        image = self.data.iloc[idx, 1:].values.astype(np.float32).reshape(1, 28, 28)
        return torch.tensor(image), torch.tensor(label)
