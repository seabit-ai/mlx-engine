import numpy as np
import torch
import mlx.core as mx
import time
from typing import Optional
import random

from mlx_engine.utils.sampling import _splitmix64


def set_seed(seed: Optional[int]) -> None:
    """
    Set the seed for all random number generators used in mlx-engine to ensure reproducible results.
    This function synchronizes the random states across multiple libraries including MLX, NumPy,
    PyTorch, and Python's built-in random module.

    Args:
        seed: The seed value to initialize random number generators. If None, a seed will be
            automatically generated using the current nanosecond timestamp. Any value in
            [0, 2**64) is accepted; MLX takes it whole, the 32-bit generators (NumPy, Python)
            get a mix of all 64 bits, so seeds that differ only above bit 31 do not collide.

    Raises:
        ValueError: If the provided seed is negative or does not fit in 64 bits.

    Returns:
        None

    Note:
        This function affects the following random number generators:
        - MLX (mx.random)
        - NumPy (np.random)
        - PyTorch (torch.manual_seed)
        - Python's built-in random module
    """
    if seed is None:
        # Get nanosecond timestamp and use it as seed
        seed = int(time.time_ns()) & (2**64 - 1)

    if seed < 0 or seed >= 2**64:
        raise ValueError("Seed must be an integer in [0, 2**64).")
    seed32 = _splitmix64(seed) >> 32

    # For MLX and MLX_LM
    mx.random.seed(seed)

    # MLX_VLM depends on numpy and torch
    np.random.seed(seed32)
    torch.manual_seed(seed)

    # Just in case
    random.seed(seed32)
