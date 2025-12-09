import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import pickle

def load_data(train_path, valid_path, test_path):
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    
    with open(valid_path, 'rb') as f:
        valid_data = pickle.load(f)

    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    
    return train_data, valid_data, test_data