"""Formato unico de checkpoints para los modelos causal y NER."""

from pathlib import Path
from typing import Any, Optional, Union

import torch


def save_checkpoint(
    path: Union[str, Path],
    *,
    model_state_dict: dict,
    tokenizer: Any,
    config: dict,
    metrics: Optional[dict] = None,
    seed: int = 42,
    extra: Optional[dict] = None,
) -> None:
    """Guarda un checkpoint autocontenido y reproducible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model_state_dict,
        "tokenizer": tokenizer,
        "config": config,
        "metrics": metrics or {},
        "seed": seed,
    }

    if extra:
        payload.update(extra)

    torch.save(payload, path)


def load_checkpoint(path: Union[str, Path], map_location: Any = "cpu") -> dict:
    """Carga checkpoints nuevos y tambien objetos Python del tokenizador."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def normalize_config(config: dict) -> dict:
    """Unifica nombres historicos como seq_len y context_size."""
    normalized = dict(config)

    if "context_size" not in normalized and "seq_len" in normalized:
        normalized["context_size"] = normalized["seq_len"]
    if "seq_len" not in normalized and "context_size" in normalized:
        normalized["seq_len"] = normalized["context_size"]
    if "lr" not in normalized and "learning_rate" in normalized:
        normalized["lr"] = normalized["learning_rate"]
    if "learning_rate" not in normalized and "lr" in normalized:
        normalized["learning_rate"] = normalized["lr"]

    return normalized
