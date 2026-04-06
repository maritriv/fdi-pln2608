"""Construccion, persistencia y validacion del indice."""

from __future__ import annotations

import hashlib
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path

import ollama

from quijote_app.config import (
    DEFAULT_CACHE_DIR,
    MIN_WORDS_PER_PASSAGE,
    PIPELINE_VERSION,
)
from quijote_app.corpus import load_corpus
from quijote_app.models import PassageIndex, Passage
from quijote_app.nlp import annotate_passages, compute_document_frequencies

# Modelo de embeddings recomendado por defecto.
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
MAX_EMBED_CHARS = 3000


class IndexError(RuntimeError):
    """Error de carga/guardado de indice."""


def default_cache_path(source: Path, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Ruta de cache determinista basada en el origen."""
    resolved_source = source.expanduser().resolve()
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", resolved_source.stem) or "corpus"
    digest = hashlib.sha1(str(resolved_source).encode("utf-8")).hexdigest()[:10]
    return cache_dir / f"index_{safe_stem}_{digest}.pkl"


def _generate_embeddings(
    passages: list[Passage], model: str = DEFAULT_EMBEDDING_MODEL
) -> None:
    """Genera y asigna embeddings a cada pasaje usando Ollama."""
    for passage in passages:
        try:
            text_for_embedding = passage.text_normalized.strip()

            if len(text_for_embedding) > MAX_EMBED_CHARS:
                print(
                    f"Advertencia: El pasaje {passage.passage_id} es demasiado largo; "
                    f"se truncará para generar el embedding."
                )
                text_for_embedding = text_for_embedding[:MAX_EMBED_CHARS]

            response = ollama.embeddings(model=model, prompt=text_for_embedding)
            passage.embedding = response.get("embedding")
        except Exception as e:
            print(
                f"Advertencia: No se pudo generar embedding para el pasaje {passage.passage_id}: {e}"
            )


def build_index(source: Path, min_words: int = MIN_WORDS_PER_PASSAGE) -> PassageIndex:
    """Construye indice desde corpus y precalcula representacion por lemas y embeddings."""
    metadata, passages = load_corpus(source=source, min_words=min_words)

    # 1. Preprocesado Clásico (Lematización, Stopwords)
    enriched_passages = annotate_passages(passages)
    lemma_document_freq = compute_document_frequencies(enriched_passages)

    # 2. Preprocesado Semántico (Vectores vía Ollama)
    print("Generando embeddings (esto puede tardar unos minutos la primera vez)...")
    _generate_embeddings(enriched_passages)

    return PassageIndex(
        metadata=metadata,
        built_at_iso=datetime.now(timezone.utc).isoformat(),
        pipeline_version=PIPELINE_VERSION,
        passages=enriched_passages,
        lemma_document_freq=lemma_document_freq,
    )


def save_index(index: PassageIndex, cache_path: Path) -> None:
    """Persistencia de indice en pickle estandar."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(index, handle)


def load_index(cache_path: Path) -> PassageIndex:
    """Carga indice desde cache."""
    try:
        with cache_path.open("rb") as handle:
            loaded = pickle.load(handle)
    except Exception as exc:  # pragma: no cover - defensivo
        raise IndexError(f"No se pudo leer la cache de indice: {cache_path}") from exc

    if not isinstance(loaded, PassageIndex):
        raise IndexError(f"Formato de cache no valido: {cache_path}")

    return loaded


def is_cache_valid(index: PassageIndex, source: Path) -> bool:
    """Valida que la cache siga sincronizada con la fuente y pipeline."""
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
    Devuelve indice listo para consulta.

    Returns:
        tuple(index, from_cache, effective_cache_path)
    """
    source = source.expanduser().resolve()
    effective_cache = cache_path or default_cache_path(source)

    if use_cache and not force_rebuild and effective_cache.exists():
        cached_index = load_index(effective_cache)

        # Validamos que la caché no solo sea válida, sino que también tenga los embeddings calculados
        has_embeddings = bool(cached_index.passages) and all(
            passage.embedding is not None for passage in cached_index.passages
        )

        if is_cache_valid(cached_index, source) and has_embeddings:
            return cached_index, True, effective_cache

    fresh_index = build_index(source=source, min_words=min_words)
    if use_cache:
        save_index(fresh_index, effective_cache)
    return fresh_index, False, effective_cache
