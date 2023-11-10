import math 
import random
from typing import List
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np



class UnNormalizer(object):
    def __init__(self, 
                 mean: List[float] = [0.485, 0.456, 0.406], 
                 std: List[float] = [0.229, 0.224, 0.225], 
                 max_val: float = 255.) -> None:
        self.mean = mean
        self.std = std
        self.max_val = max_val

    def __call__(self, tensor: torch.tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s * self.max_val).add_(m * self.max_val)
        return tensor
