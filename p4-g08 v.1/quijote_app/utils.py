"""Utilidades de normalizacion y rendering en terminal."""

from __future__ import annotations

import re
import unicodedata

from quijote_app.config import SNIPPET_MAX_CHARS

WORD_RE = re.compile(r"\w+", flags=re.IGNORECASE)


def collapse_spaces(text: str) -> str:
    """Colapsa espacios en blanco consecutivos."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    """Normaliza un texto para comparaciones robustas."""
    if not text:
        return ""

    decomposed = unicodedata.normalize("NFD", text.lower())
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    without_marks = without_marks.replace("_", " ")
    cleaned = re.sub(r"[^\w]+", " ", without_marks, flags=re.UNICODE)
    return collapse_spaces(cleaned)


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

    for token in normalized_query.split():
        if not token:
            continue
        token_pattern = re.compile(
            rf"(?<!\w){re.escape(token)}(?!\w)",
            flags=re.IGNORECASE,
        )
        token_match = token_pattern.search(normalized_text)
        if token_match:
            return token_match.start(), token_match.end()

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
