"""Reproducibility helper."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, seed_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
