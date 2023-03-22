import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MNISTDataset(dataset):
        def __init__(self, data):
                self.data = data
        
        def __len__(self):
                return len(self.data)
        
        def __getItem__(self, idx):
                label = self.data.iloc[idx, 0]
                image = self.data.iloc[idx, 1].values.astype(np.float32).reshape(1, 28, 28)
                return torch.tensor(image), torch.tensor(label)
        

def load_data(train_csv, test_csv):
        train_data = pd.read_csv(train_data)
        test_data = pd.read_csv(test_data)

        return train_data, test_data


