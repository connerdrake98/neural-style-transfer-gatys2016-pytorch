"""
Utilities: device selection and seeding for reproducibility.

"get_device" prefers CUDA if available, otherwise fall back to Apple's
Metal Performance Shaders (MPS) when available, otherwise use CPU; for convenience on laptops with MPS or machines with NVIDIA GPUs

"set_seed" sets seeds for Python random, NumPy, PyTorch. 
Making it reproducible on GPU can require additional flags (deterministic CuDNN settings), might slow execution; using a baseline for now
"""

import torch
import random
import numpy as np


def get_device(prefer_cuda=True):
    """
    Return the best device available

    If "prefer_cuda" and CUDA is available, return a CUDA device
    Otherwise try to use MPS (Apple Silicon) before CPU fallback
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 1234):
    """
    Set random seeds for reproducibility (Python random, NumPy, PyTorch)

    Also sets CUDA manual seed on CUDA devices
    Full bitwise reproducibility on GPU can require additional flags which are not set here
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
