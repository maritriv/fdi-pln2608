"""Configuración central del proyecto."""

from pathlib import Path

PIPELINE_VERSION = "1.0.0"

DEFAULT_LIMIT = 5
MIN_WORDS_PER_PASSAGE = 12
SNIPPET_MAX_CHARS = 340

DEFAULT_CACHE_DIR = Path("cache")

DEFAULT_SOURCE_CANDIDATES = (
    Path("data/El_ingenioso_hidalgo_don_Quijote_de_la_Mancha.zip"),
    Path("El_ingenioso_hidalgo_don_Quijote_de_la_Mancha.zip"),
    Path("data/2000-h.htm"),
    Path("2000-h.htm"),
)

