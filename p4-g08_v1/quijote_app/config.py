"""Configuracion central del proyecto."""

from pathlib import Path

PIPELINE_VERSION = "1.0"

DEFAULT_LIMIT = 5
MIN_WORDS_PER_PASSAGE = 45
MIN_WORDS_PER_BLOCK = 8
CHUNK_TARGET_WORDS = 120
CHUNK_OVERLAP_WORDS = 35
SNIPPET_MAX_CHARS = 340

DEFAULT_CACHE_DIR = Path("cache")

# --- MODELOS DE IA ---
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "llama3"  # Cambia esto si el profesor exige otro modelo

DEFAULT_SOURCE_CANDIDATES = (
    Path("data/El_ingenioso_hidalgo_don_Quijote_de_la_Mancha.zip"),
    Path("El_ingenioso_hidalgo_don_Quijote_de_la_Mancha.zip"),
    Path("data/2000-h.htm"),
    Path("2000-h.htm"),
)