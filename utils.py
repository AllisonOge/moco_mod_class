from typing import List, Any
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import csv
from pathlib import Path


def real2complex(signal):
    assert signal.shape[0] == 2, "Signal must have 2 rows"
    return signal[0] + 1j*signal[1]


def spectrogram_plot(signal, metadata):
    fs = metadata.get("effective_sample_rate", 1.0)
    f, t, Sxx = spectrogram(signal, fs, nperseg=1024,
                            noverlap=512, return_onesided=False)

    # Sort the frequencies and spectrogram rows
    sorted_indices = np.argsort(f)
    f = f[sorted_indices]
    Sxx = Sxx[sorted_indices, :]

    plt.pcolormesh(t, f / 1e6, 10*np.log10(np.abs(Sxx)))
    plt.ylabel('Frequency [MHz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Power [dB]')
    plt.show()


# ===========training utils================
def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: nn.Module, device: str):
    model.train()
    running_loss = 0.0

    for i, (iq_data, label) in enumerate(train_loader):
        iq_data, label = iq_data.to(device), label.to(device)

        optimizer.zero_grad()

        output = model(iq_data)
        loss = criterion(output, label[:, 0].long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def validate_epoch(model: nn.Module,
                   val_loader: DataLoader,
                   criterion: nn.Module, device: str):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (iq_data, label) in enumerate(val_loader):
            iq_data, label = iq_data.to(device), label.to(device)

            output = model(iq_data)
            loss = criterion(output, label[:, 0].long())
            running_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label[:, 0].long()).sum().item()

    return running_loss / len(val_loader), correct / total


class Tracker:
    def __init__(self, metric, mode='auto'):
        self.metric = metric
        self.mode = mode
        self.mode_dict = {
            'auto': np.less if 'loss' in metric else np.greater,
            'min': np.less,
            'max': np.greater
        }
        self.operator = self.mode_dict[mode]

        self._best = np.inf if 'loss' in metric else -np.inf

    @property
    def best(self):
        return self._best

    @best.setter
    def best(self, value):
        self._best = value


class CSVLogger:
    def __init__(self, sep=",", filename="results.csv", append=False):
        self.sep = sep
        self.filename = Path(filename)
        self.append = append
        self.writer = None
        self.file = None

    def _save_stats(self, stats: List[Any]):
        class CustomDialet(csv.excel):
            delimiter = self.sep

        if not self.writer:
            self.writer = csv.writer(self.file, dialect=CustomDialet)

        self.writer.writerow(stats)
        self.file.flush()

    def _init_file(self):
        self.file = open(self.filename, "a+" if self.append else "w+")
        self.file.seek(0)
        if not self.append or not bool(len(self.file.read())):
            header = ["epoch", "train_loss", "val_loss", "val_acc", "lr"]
            # only add the header if the file is empty
            self._save_stats(header)

    def save(self, stats: List[Any]):
        if not self.file:
            self._init_file()
        self._save_stats(stats)

    def read(self):
        if not self.file:
            self._init_file()
        self.file.seek(0)
        reader = csv.reader(self.file, delimiter=self.sep)
        return list(reader)

    def close(self):
        if self.file:
            self.file.close()
        self.writer = None
