"""Modelos de datos para corpus, indice y resultados."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Passage:
    """Unidad recuperable del corpus."""

    passage_id: str
    order: int
    chapter: str | None
    part: str | None
    text_original: str
    text_normalized: str
    lemmas: tuple[str, ...] = field(default_factory=tuple)
    content_lemmas: tuple[str, ...] = field(default_factory=tuple)
    embedding: list[float] | None = None  # ¡Nuevo! Representación vectorial del pasaje


@dataclass
class CorpusMetadata:
    """Metadatos del corpus que permiten validar cache."""

    source_path: Path
    source_kind: str
    source_size: int
    source_mtime_ns: int
    selected_entry: str | None
    chapter_count: int
    part_count: int
    passage_count: int


@dataclass
class PassageIndex:
    """Indice de pasajes del corpus."""

    metadata: CorpusMetadata
    built_at_iso: str
    pipeline_version: str
    passages: list[Passage] = field(default_factory=list)
    lemma_document_freq: dict[str, int] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Resultado de recuperacion con metrica de relevancia."""

    passage: Passage
    score: float
    exact_matches: int
    matched_terms: int
    total_term_hits: int
