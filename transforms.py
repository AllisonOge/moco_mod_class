"""
Implement augmentation transforms for the dataset.
"""
import torch
import scipy.signal as signal
import numpy as np


class NoOp:
    """
    No operation transform.

    :rtype: torch.Tensor

    Example:
        >>> x = torch.arange(1, 51).reshape(10, 5)
        >>> transform = NoOp()
        >>> transform(x)
    """

    def __call__(self, x):
        return x


class RandomZeroMasking:
    """
    Mask a random number of elements in the input tensor with zeros.

    :param max_rate: Maximum mask rate.
    :type max_rate: int (default = 5)
    :param dim: Dimension along which to mask the elements.
    :type dim: int (default = -1)
    :rtype: torch.Tensor

    """

    def __init__(self, max_rate=.1, dim=-1):
        self.max_rate = max_rate
        self.dim = dim

    def __call__(self, x: torch.Tensor):
        mask_size = torch.randint(
            0, int(x.size(self.dim)*self.max_rate) + 1, (1,)).item()
        mask = torch.ones_like(x)
        mask_idx = [slice(None)] * x.ndim
        rand_start_idxs = torch.randint(
            0, x.size(self.dim), (max(min((x.size(self.dim) - mask_size) // 5, 5), 1),))
        mask_idx[self.dim] = torch.flatten(rand_start_idxs.unsqueeze(1)
                                           + torch.arange(mask_size).unsqueeze(0))
        mask_idx[self.dim] = mask_idx[self.dim].clamp(0, x.size(self.dim) - 1)
        mask[mask_idx] = 0
        return x * mask
