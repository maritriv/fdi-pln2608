"""Búsqueda y ranking de pasajes."""

from __future__ import annotations

import re

from quijote_app.models import Passage, PassageIndex, SearchResult
from quijote_app.utils import expand_term_variants, normalize_text, tokenize_normalized


def search_passages(
    index: PassageIndex,
    query: str,
    limit: int,
    chapter_filter: str | None = None,
) -> list[SearchResult]:
    """Busca pasajes por término o frase y ordena por relevancia."""
    normalized_query = normalize_text(query)
    if not normalized_query:
        raise ValueError("La consulta está vacía tras normalización.")

    terms = sorted(set(tokenize_normalized(normalized_query)))
    if not terms:
        raise ValueError("La consulta debe contener al menos un término útil.")

    term_patterns: dict[str, re.Pattern[str]] = {}
    for term in terms:
        variants = sorted(expand_term_variants(term), key=len, reverse=True)
        joined = "|".join(re.escape(variant) for variant in variants)
        term_patterns[term] = re.compile(
            rf"(?<!\w)(?:{joined})(?!\w)",
            flags=re.IGNORECASE,
        )

    chapter_filter_normalized = normalize_text(chapter_filter) if chapter_filter else None
    results: list[SearchResult] = []

    for passage in index.passages:
        if chapter_filter_normalized:
            chapter_normalized = normalize_text(passage.chapter or "")
            if chapter_filter_normalized not in chapter_normalized:
                continue

        scored = _score_passage(
            passage=passage,
            normalized_query=normalized_query,
            term_patterns=term_patterns,
        )
        if scored is not None:
            results.append(scored)

    results.sort(key=lambda item: (-item.score, item.passage.order))
    return results[:limit]


def _score_passage(
    passage: Passage,
    normalized_query: str,
    term_patterns: dict[str, re.Pattern[str]],
) -> SearchResult | None:
    text = passage.text_normalized
    exact_matches = _count_exact_query_matches(text, normalized_query)

    matched_terms = 0
    total_term_hits = 0
    for pattern in term_patterns.values():
        hits = len(pattern.findall(text))
        total_term_hits += hits
        if hits > 0:
            matched_terms += 1

    if exact_matches == 0 and total_term_hits == 0:
        return None

    term_count = max(1, len(term_patterns))
    coverage = matched_terms / term_count
    length_words = max(1, len(text.split()))

    exact_capped = min(exact_matches, 6)
    term_hits_capped = min(total_term_hits, 12)

    score = 0.0
    if exact_matches > 0:
        score += 10.0 + exact_capped * 2.5

    score += coverage * 10.0
    score += term_hits_capped * 1.4

    if matched_terms == term_count:
        score += 4.0

    if length_words > 160:
        score -= min(12.0, (length_words - 160) * 0.08)
    elif length_words < 25:
        score -= 1.0
    else:
        score += 1.0

    density = term_hits_capped / length_words
    score += min(4.0, density * 40.0)

    return SearchResult(
        passage=passage,
        score=score,
        exact_matches=exact_matches,
        matched_terms=matched_terms,
        total_term_hits=total_term_hits,
    )


def _count_exact_query_matches(text: str, normalized_query: str) -> int:
    """Cuenta coincidencias exactas de la consulta completa respetando l?mites de palabra."""
    exact_pattern = re.compile(
        rf"(?<!\w){re.escape(normalized_query)}(?!\w)",
        flags=re.IGNORECASE,
    )
    return len(exact_pattern.findall(text))
