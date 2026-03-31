"""Construcción, persistencia y validación del índice."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib

from quijote_app.config import DEFAULT_CACHE_DIR, MIN_WORDS_PER_PASSAGE, PIPELINE_VERSION
from quijote_app.corpus import load_corpus
from quijote_app.models import PassageIndex


class IndexError(RuntimeError):
    """Error de carga/guardado de índice."""


def default_cache_path(source: Path, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Ruta de caché determinista basada en el origen."""
    resolved_source = source.expanduser().resolve()
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", resolved_source.stem) or "corpus"
    digest = hashlib.sha1(str(resolved_source).encode("utf-8")).hexdigest()[:10]
    return cache_dir / f"index_{safe_stem}_{digest}.joblib"


def build_index(source: Path, min_words: int = MIN_WORDS_PER_PASSAGE) -> PassageIndex:
    """Construye índice desde corpus."""
    metadata, passages = load_corpus(source=source, min_words=min_words)
    return PassageIndex(
        metadata=metadata,
        built_at_iso=datetime.now(timezone.utc).isoformat(),
        pipeline_version=PIPELINE_VERSION,
        passages=passages,
    )


def save_index(index: PassageIndex, cache_path: Path) -> None:
    """Persistencia de índice."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, cache_path)


def load_index(cache_path: Path) -> PassageIndex:
    """Carga índice desde caché."""
    try:
        return joblib.load(cache_path)
    except Exception as exc:  # pragma: no cover - defensivo
        raise IndexError(f"No se pudo leer la caché de índice: {cache_path}") from exc


def is_cache_valid(index: PassageIndex, source: Path) -> bool:
    """Valida que la caché siga sincronizada con la fuente y pipeline."""
    source = source.expanduser().resolve()
    if not source.exists():
        return False

    meta = index.metadata
    if index.pipeline_version != PIPELINE_VERSION:
        return False
    if meta.source_path.expanduser().resolve() != source:
        return False
    if meta.source_size != source.stat().st_size:
        return False
    if meta.source_mtime_ns != source.stat().st_mtime_ns:
        return False
    return True


def load_or_build_index(
    source: Path,
    cache_path: Path | None = None,
    use_cache: bool = True,
    force_rebuild: bool = False,
    min_words: int = MIN_WORDS_PER_PASSAGE,
) -> tuple[PassageIndex, bool, Path]:
    """
    Devuelve índice listo para consulta.

    Returns:
        tuple(index, from_cache, effective_cache_path)
    """
    source = source.expanduser().resolve()
    effective_cache = cache_path or default_cache_path(source)

    if use_cache and not force_rebuild and effective_cache.exists():
        cached_index = load_index(effective_cache)
        if is_cache_valid(cached_index, source):
            return cached_index, True, effective_cache

    fresh_index = build_index(source=source, min_words=min_words)
    if use_cache:
        save_index(fresh_index, effective_cache)
    return fresh_index, False, effective_cache

