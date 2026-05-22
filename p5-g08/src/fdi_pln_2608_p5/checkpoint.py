"""Formato unico de checkpoints para los modelos causal y NER."""

import importlib
import sys
import types
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
    _install_legacy_pickle_aliases()
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _install_legacy_pickle_aliases() -> None:
    """Permite leer checkpoints antiguos sin mantener el paquete modules/."""

    legacy_package = "fdi_pln_2608_p5.modules"
    if legacy_package not in sys.modules:
        sys.modules[legacy_package] = types.ModuleType(legacy_package)

    aliases = {
        "attention": "fdi_pln_2608_p5.model.attention",
        "data": "fdi_pln_2608_p5.data.dataset",
        "generate": "fdi_pln_2608_p5.generation.generate",
        "model": "fdi_pln_2608_p5.model.transformer",
        "ner": "fdi_pln_2608_p5.model.ner",
        "ner_predict": "fdi_pln_2608_p5.generation.ner_predict",
        "tokenizer": "fdi_pln_2608_p5.tokenizer",
        "train": "fdi_pln_2608_p5.training.train_causal",
        "train_ner": "fdi_pln_2608_p5.training.train_ner",
    }

    package = sys.modules[legacy_package]
    for legacy_name, target_name in aliases.items():
        module = importlib.import_module(target_name)
        sys.modules[f"{legacy_package}.{legacy_name}"] = module
        setattr(package, legacy_name, module)


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
