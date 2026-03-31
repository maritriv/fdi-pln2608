"""Busqueda y ranking de pasajes."""

from __future__ import annotations

from collections import Counter
import re
from math import log

from quijote_app.models import Passage, PassageIndex, SearchResult
from quijote_app.utils import normalize_text, stem_spanish_token, tokenize_normalized, tokens_match_by_root

STOPWORDS_ES = {
    "a",
    "al",
    "con",
    "de",
    "del",
    "el",
    "en",
    "la",
    "las",
    "lo",
    "los",
    "por",
    "para",
    "que",
    "se",
    "un",
    "una",
    "unos",
    "unas",
    "y",
}


def search_passages(
    index: PassageIndex,
    query: str,
    limit: int,
    chapter_filter: str | None = None,
) -> list[SearchResult]:
    """Busca pasajes por termino o frase y ordena por relevancia."""
    normalized_query = normalize_text(query)
    if not normalized_query:
        raise ValueError("La consulta esta vacia tras normalizacion.")

    query_tokens = tokenize_normalized(normalized_query)
    if not query_tokens:
        raise ValueError("La consulta debe contener al menos un termino util.")

    content_tokens = _select_content_query_tokens(query_tokens)
    effective_tokens = content_tokens if content_tokens else query_tokens
    normalized_effective_query = " ".join(effective_tokens)

    query_term_counts = Counter(effective_tokens)
    total_query_terms = max(1, sum(query_term_counts.values()))

    term_profiles = {
        term: {
            "stem": stem_spanish_token(term),
            "query_tf": count / total_query_terms,
            "idf": 1.0,
        }
        for term, count in query_term_counts.items()
    }

    _apply_idf(index=index, term_profiles=term_profiles)

    chapter_filter_normalized = normalize_text(chapter_filter) if chapter_filter else None
    results: list[SearchResult] = []

    for passage in index.passages:
        if chapter_filter_normalized:
            chapter_normalized = normalize_text(passage.chapter or "")
            if chapter_filter_normalized not in chapter_normalized:
                continue

        scored = _score_passage(
            passage=passage,
            normalized_full_query=normalized_query,
            normalized_effective_query=normalized_effective_query,
            term_profiles=term_profiles,
        )
        if scored is not None:
            results.append(scored)

    results.sort(key=lambda item: (-item.score, item.passage.order))
    return results[:limit]


def _select_content_query_tokens(query_tokens: list[str]) -> list[str]:
    return [token for token in query_tokens if token not in STOPWORDS_ES]


def _score_passage(
    passage: Passage,
    normalized_full_query: str,
    normalized_effective_query: str,
    term_profiles: dict[str, dict[str, float | str]],
) -> SearchResult | None:
    text = passage.text_normalized
    exact_matches = max(
        _count_exact_query_matches(text, normalized_full_query),
        _count_exact_query_matches(text, normalized_effective_query),
    )

    passage_tokens = tokenize_normalized(text)
    passage_token_stems = [stem_spanish_token(token) for token in passage_tokens]
    doc_length = max(1, len(passage_tokens))

    matched_terms = 0
    total_term_hits = 0
    tfidf_score = 0.0
    positions_by_term: dict[str, list[int]] = {}

    for term, profile in term_profiles.items():
        hits, positions = _count_term_hits(
            query_term=term,
            query_stem=str(profile["stem"]),
            passage_tokens=passage_tokens,
            passage_token_stems=passage_token_stems,
        )

        positions_by_term[term] = positions
        total_term_hits += hits

        if hits > 0:
            matched_terms += 1

        tf = hits / doc_length
        idf = float(profile["idf"])
        query_tf = float(profile["query_tf"])
        tfidf_score += tf * idf * query_tf

    if exact_matches == 0 and total_term_hits == 0:
        return None

    term_count = max(1, len(term_profiles))
    coverage = matched_terms / term_count

    score = tfidf_score * 120.0
    score += coverage * 12.0

    if term_count > 1 and coverage >= 0.999:
        score += 4.0

    if exact_matches > 0:
        score += 4.0 + min(6, exact_matches) * 2.0

    proximity_span = _minimum_cover_span(positions_by_term)
    if proximity_span is not None:
        compactness = max(0.0, min(1.0, matched_terms / proximity_span))
        score += compactness * 5.0

    length_words = max(1, len(text.split()))
    if length_words > 180:
        score -= min(10.0, (length_words - 180) * 0.07)

    return SearchResult(
        passage=passage,
        score=score,
        exact_matches=exact_matches,
        matched_terms=matched_terms,
        total_term_hits=total_term_hits,
    )


def _count_term_hits(
    query_term: str,
    query_stem: str,
    passage_tokens: list[str],
    passage_token_stems: list[str],
) -> tuple[int, list[int]]:
    positions: list[int] = []

    for idx, (token, token_stem) in enumerate(zip(passage_tokens, passage_token_stems)):
        if tokens_match_by_root(query_term, token):
            positions.append(idx)
            continue

        if len(query_stem) >= 3 and query_stem == token_stem:
            positions.append(idx)

    return len(positions), positions


def _minimum_cover_span(positions_by_term: dict[str, list[int]]) -> int | None:
    matched_terms = {
        term: positions
        for term, positions in positions_by_term.items()
        if positions
    }
    if len(matched_terms) < 2:
        return None

    merged: list[tuple[int, str]] = []
    for term, positions in matched_terms.items():
        merged.extend((position, term) for position in positions)
    merged.sort(key=lambda item: item[0])

    needed = len(matched_terms)
    covered = 0
    counts: dict[str, int] = {}
    left = 0
    best_span: int | None = None

    for right, (right_pos, right_term) in enumerate(merged):
        prev = counts.get(right_term, 0)
        counts[right_term] = prev + 1
        if prev == 0:
            covered += 1

        while covered == needed and left <= right:
            left_pos, left_term = merged[left]
            span = right_pos - left_pos + 1
            if best_span is None or span < best_span:
                best_span = span

            counts[left_term] -= 1
            if counts[left_term] == 0:
                covered -= 1
            left += 1

    return best_span


def _apply_idf(
    index: PassageIndex,
    term_profiles: dict[str, dict[str, float | str]],
) -> None:
    if not term_profiles:
        return

    total_docs = max(1, len(index.passages))
    document_freq = {term: 0 for term in term_profiles}

    for passage in index.passages:
        passage_tokens = tokenize_normalized(passage.text_normalized)
        passage_token_stems = [stem_spanish_token(token) for token in passage_tokens]

        for term, profile in term_profiles.items():
            hits, _ = _count_term_hits(
                query_term=term,
                query_stem=str(profile["stem"]),
                passage_tokens=passage_tokens,
                passage_token_stems=passage_token_stems,
            )
            if hits > 0:
                document_freq[term] += 1

    for term, profile in term_profiles.items():
        df = document_freq[term]
        profile["idf"] = log((total_docs + 1) / (df + 1)) + 1.0


def _count_exact_query_matches(text: str, normalized_query: str) -> int:
    """Cuenta coincidencias exactas de la consulta completa con limites de palabra."""
    if not normalized_query:
        return 0

    exact_pattern = re.compile(
        rf"(?<!\w){re.escape(normalized_query)}(?!\w)",
        flags=re.IGNORECASE,
    )
    return len(exact_pattern.findall(text))
