"""
A python class for the Dataset (R22)
"""
import h5py
from torch.utils.data import Dataset


class MoCo_H5_Dataset(Dataset):
    def __init__(self, data_h5_path, label='data', data_txfm=None):
        self.data_h5_path = data_h5_path
        self.label = label
        self.data_txfm = data_txfm

    def __len__(self):
        with h5py.File(self.data_h5_path, 'r') as f:
            length = len(f[self.label])
        return length

    def __getitem__(self, idx):
        with h5py.File(self.data_h5_path, 'r') as f:
            iq_data = f[self.label][idx]
        if self.data_txfm:
            iq_data = self.data_txfm(iq_data)
        return iq_data


class RML22_Dataset(Dataset):
    def __init__(self, data_h5_path, iqlabel="data", label="label", data_txfm=None):
        self.data_h5_path = data_h5_path
        self.iqlabel = iqlabel
        self.label = label
        self.data_txfm = data_txfm

    def __len__(self):
        with h5py.File(self.data_h5_path, 'r') as f:
            length = len(f[self.iqlabel])
        return length

    def __getitem__(self, idx):
        with h5py.File(self.data_h5_path, 'r') as f:
            iq_data = f[self.iqlabel][idx]
            label = f[self.label][idx]
        if self.data_txfm:
            iq_data = self.data_txfm(iq_data)
        return iq_data, label
