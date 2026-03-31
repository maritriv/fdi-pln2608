"""Utilidades de normalizacion y rendering en terminal."""

from __future__ import annotations

from functools import lru_cache
import re
import unicodedata

from quijote_app.config import SNIPPET_MAX_CHARS

WORD_RE = re.compile(r"\w+", flags=re.IGNORECASE)
SPACY_MODEL_NAME = "es_core_news_sm"

try:
    import spacy
except ImportError:  # pragma: no cover - fallback defensivo
    spacy = None  # type: ignore[assignment]

# Sufijos frecuentes (sustantivos/adjetivos/verbos) para una raiz aproximada.
# Se aplican de mayor a menor longitud y solo si la raiz restante tiene >= 3 letras.
STEM_SUFFIXES = (
    "ariamos",
    "eriamos",
    "iriamos",
    "ariais",
    "eriais",
    "iriais",
    "aremos",
    "eremos",
    "iremos",
    "arian",
    "erian",
    "irian",
    "arias",
    "erias",
    "irias",
    "asteis",
    "isteis",
    "abamos",
    "iamos",
    "aramos",
    "eramos",
    "iramos",
    "amiento",
    "imientos",
    "imiento",
    "aciones",
    "adores",
    "adoras",
    "logias",
    "ucion",
    "uciones",
    "mente",
    "idades",
    "ismos",
    "istas",
    "anzas",
    "ancia",
    "encias",
    "adora",
    "ador",
    "acion",
    "able",
    "ible",
    "ando",
    "iendo",
    "yendo",
    "ados",
    "adas",
    "idos",
    "idas",
    "aban",
    "abas",
    "aran",
    "eran",
    "iran",
    "aron",
    "ieron",
    "amos",
    "emos",
    "imos",
    "ais",
    "eis",
    "aba",
    "ada",
    "ado",
    "ida",
    "ido",
    "ias",
    "ian",
    "aria",
    "eria",
    "iria",
    "ar",
    "er",
    "ir",
)


def collapse_spaces(text: str) -> str:
    """Colapsa espacios en blanco consecutivos."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    """Normaliza un texto para busqueda robusta."""
    if not text:
        return ""

    decomposed = unicodedata.normalize("NFD", text.lower())
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    without_marks = without_marks.replace("_", " ")
    cleaned = re.sub(r"[^\w]+", " ", without_marks, flags=re.UNICODE)
    return collapse_spaces(cleaned)


def tokenize_normalized(text: str) -> list[str]:
    """Tokeniza usando spaCy (es) sobre texto normalizado."""
    normalized = normalize_text(text)
    if not normalized:
        return []

    if spacy is None:
        return normalized.split()

    doc = _get_spanish_tokenizer().make_doc(normalized)
    return [token.text for token in doc if not token.is_space]


@lru_cache(maxsize=1)
def _get_spanish_tokenizer():
    """Instancia perezosa de spaCy para tokenizar en espanol."""
    try:
        return spacy.load(
            SPACY_MODEL_NAME,
            exclude=[
                "attribute_ruler",
                "lemmatizer",
                "morphologizer",
                "ner",
                "parser",
                "senter",
                "tagger",
                "tok2vec",
            ],
        )
    except Exception:
        # Fallback robusto si el modelo no esta instalado.
        return spacy.blank("es")


def expand_term_variants(term: str) -> set[str]:
    """Genera variantes singular/plural basicas para un termino."""
    normalized = normalize_text(term)
    if not normalized:
        return set()
    if " " in normalized:
        return {normalized}

    variants = {normalized}
    pending = [normalized]
    seen: set[str] = set()

    while pending:
        candidate = pending.pop()
        if candidate in seen:
            continue
        seen.add(candidate)

        plural = _to_plural_variant(candidate)
        if plural and plural not in variants:
            variants.add(plural)
            pending.append(plural)

        for singular in _singular_variants(candidate):
            if singular not in variants:
                variants.add(singular)
                pending.append(singular)

    return {item for item in variants if item and " " not in item and len(item) >= 2}


def _to_plural_variant(term: str) -> str | None:
    if len(term) < 2:
        return None
    if term.endswith("z"):
        return term[:-1] + "ces"
    if term.endswith("s"):
        return None
    if term[-1] in "aeiou":
        return term + "s"
    return term + "es"


def _singular_variants(term: str) -> set[str]:
    variants: set[str] = set()
    if len(term) < 3:
        return variants

    if term.endswith("ces") and len(term) > 3:
        variants.add(term[:-3] + "z")

    if term.endswith("s") and len(term) > 4 and term[-2] in "aeiou":
        variants.add(term[:-1])

    if term.endswith("es") and len(term) > 5 and term[-3] not in "aeiou":
        variants.add(term[:-2])

    return {item for item in variants if item}


def stem_spanish_token(token: str) -> str:
    """Calcula una raiz aproximada para comparar variantes morfologicas."""
    normalized = normalize_text(token)
    if not normalized or " " in normalized:
        return normalized

    root = normalized

    singulars = sorted(_singular_variants(root), key=len)
    if singulars:
        root = singulars[0]

    for suffix in STEM_SUFFIXES:
        if root.endswith(suffix) and len(root) - len(suffix) >= 3:
            root = root[: -len(suffix)]
            break

    return root


def tokens_match_by_root(query_token: str, candidate_token: str) -> bool:
    """True si dos tokens coinciden por forma o por raiz."""
    normalized_query = normalize_text(query_token)
    normalized_candidate = normalize_text(candidate_token)

    if not normalized_query or not normalized_candidate:
        return False
    if " " in normalized_query or " " in normalized_candidate:
        return False

    if normalized_candidate in expand_term_variants(normalized_query):
        return True

    query_stem = stem_spanish_token(normalized_query)
    candidate_stem = stem_spanish_token(normalized_candidate)
    if len(query_stem) < 3 or len(candidate_stem) < 3:
        return False

    return query_stem == candidate_stem


def count_substring_occurrences(text: str, needle: str) -> int:
    """Cuenta ocurrencias no solapadas de `needle` en `text`."""
    if not needle:
        return 0
    return text.count(needle)


def normalize_with_mapping(text: str) -> tuple[str, list[int]]:
    """Normaliza y devuelve mapeo de indice normalizado -> indice original."""
    normalized_chars: list[str] = []
    mapping: list[int] = []
    prev_space = True

    for idx, char in enumerate(text):
        pieces = unicodedata.normalize("NFD", char.lower())
        for piece in pieces:
            if unicodedata.category(piece) == "Mn":
                continue
            if piece.isalnum():
                normalized_chars.append(piece)
                mapping.append(idx)
                prev_space = False
            elif not prev_space:
                normalized_chars.append(" ")
                mapping.append(idx)
                prev_space = True

    if normalized_chars and normalized_chars[-1] == " ":
        normalized_chars.pop()
        mapping.pop()

    return "".join(normalized_chars), mapping


def find_query_span(text: str, query: str) -> tuple[int, int] | None:
    """Encuentra la mejor coincidencia de `query` en `text`."""
    normalized_query = normalize_text(query)
    if not normalized_query:
        return None

    normalized_text, mapping = normalize_with_mapping(text)
    if not normalized_text:
        return None

    normalized_match = _find_normalized_match_span(normalized_text, normalized_query)
    if normalized_match is None:
        return None

    position, end_exclusive = normalized_match
    end_position = end_exclusive - 1
    if end_position >= len(mapping):
        return None

    return mapping[position], mapping[end_position] + 1


def _find_normalized_match_span(normalized_text: str, normalized_query: str) -> tuple[int, int] | None:
    exact_pattern = re.compile(
        rf"(?<!\w){re.escape(normalized_query)}(?!\w)",
        flags=re.IGNORECASE,
    )
    exact_match = exact_pattern.search(normalized_text)
    if exact_match:
        return exact_match.start(), exact_match.end()

    if " " not in normalized_query:
        return _find_token_match_in_normalized_text(normalized_text, normalized_query)

    for token in tokenize_normalized(normalized_query):
        token_match = _find_token_match_in_normalized_text(normalized_text, token)
        if token_match is not None:
            return token_match

    return None


def _find_token_match_in_normalized_text(normalized_text: str, token: str) -> tuple[int, int] | None:
    variants = sorted(expand_term_variants(token), key=len, reverse=True)
    for variant in variants:
        variant_pattern = re.compile(
            rf"(?<!\w){re.escape(variant)}(?!\w)",
            flags=re.IGNORECASE,
        )
        variant_match = variant_pattern.search(normalized_text)
        if variant_match:
            return variant_match.start(), variant_match.end()

    query_stem = stem_spanish_token(token)
    if len(query_stem) < 3:
        return None

    for candidate in WORD_RE.finditer(normalized_text):
        if stem_spanish_token(candidate.group(0)) == query_stem:
            return candidate.start(), candidate.end()

    return None


def highlight_span(text: str, span: tuple[int, int] | None) -> str:
    """Resalta un tramo en un texto usando corchetes."""
    if span is None:
        return text

    start, end = span
    if start < 0 or end > len(text) or start >= end:
        return text

    return f"{text[:start]}[{text[start:end]}]{text[end:]}"


def render_excerpt(text: str, query: str, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    """Renderiza un pasaje con recorte contextual y resaltado."""
    span = find_query_span(text, query)
    source_length = len(text)

    if source_length <= max_chars:
        excerpt = text
        local_span = span
        start = 0
        end = source_length
    else:
        midpoint = ((span[0] + span[1]) // 2) if span else source_length // 2
        start = max(0, midpoint - max_chars // 2)
        end = min(source_length, start + max_chars)
        start = max(0, end - max_chars)

        excerpt = text[start:end]
        local_span = None
        if span and span[1] > start and span[0] < end:
            local_start = max(0, span[0] - start)
            local_end = min(len(excerpt), span[1] - start)
            if local_start < local_end:
                local_span = (local_start, local_end)

    highlighted = highlight_span(excerpt, local_span)

    prefix = "... " if start > 0 else ""
    suffix = " ..." if end < source_length else ""
    return f"{prefix}{highlighted}{suffix}"
