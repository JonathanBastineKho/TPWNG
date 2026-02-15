import torch
from torch import nn
from torch.utils.data import Dataset
from typing import Optional

class UCFCrimeDataset(Dataset):
    def __init__(self):
        super().__init__()