import torch
import scipy
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset

class DynDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, dtype=torch.float32):
        super(DynDataset, self).__init__()

        self.root_dir = root_dir
        self.dtype = dtype
        mat_contents = sio.loadmat(self.root_dir)
        self.t = torch.tensor(mat_contents['t'], dtype=self.dtype)
        self.x = torch.tensor(mat_contents['x'], dtype=self.dtype)
        self.u = torch.tensor(mat_contents['u'], dtype=self.dtype)
        self.x_dot = torch.tensor(mat_contents['x_dot'], dtype=self.dtype)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        t = self.t[idx,:]
        x = self.x[idx,:]
        u = self.u[idx,:]
        x_dot = self.x_dot[idx,:]

        return t, x, u, x_dot