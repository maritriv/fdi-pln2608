"""Utilidades compartidas para reproducibilidad y ejecucion."""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fija las semillas usadas por Python, NumPy y PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Devuelve un dispositivo PyTorch, usando CPU por defecto en la practica."""
    if device is not None:
        return torch.device(device)
    return torch.device("cpu")
