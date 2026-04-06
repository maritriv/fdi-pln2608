"""Busqueda clasica por lemas con ranking TF-IDF y Busqueda Semantica."""

from __future__ import annotations

from collections import Counter
from math import log, sqrt
import re
import ollama

from quijote_app.models import Passage, PassageIndex, SearchResult
from quijote_app.nlp import select_query_terms
from quijote_app.utils import normalize_text

# --- BÚSQUEDA CLÁSICA ---

def search_passages(
    index: PassageIndex,
    query: str,
    limit: int,
    chapter_filter: str | None = None,
) -> list[SearchResult]:
    """Busca pasajes por lemas y ordena por relevancia."""
    normalized_query = normalize_text(query)
    if not normalized_query:
        raise ValueError("La consulta esta vacia tras normalizacion.")

    query_lemmas, effective_terms = select_query_terms(query)
    if not effective_terms:
        raise ValueError("La consulta debe contener al menos un termino util.")

    query_term_counts = Counter(effective_terms)
    total_query_terms = max(1, sum(query_term_counts.values()))

    term_profiles = {
        term: {
            "query_tf": count / total_query_terms,
            "idf": _idf(index=index, term=term),
        }
        for term, count in query_term_counts.items()
    }

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
            query_lemmas=query_lemmas,
            effective_terms=effective_terms,
            term_profiles=term_profiles,
        )
        if scored is not None:
            results.append(scored)

    results.sort(key=lambda item: (-item.score, item.passage.order))
    return results[:limit]


def _score_passage(
    passage: Passage,
    normalized_full_query: str,
    query_lemmas: list[str],
    effective_terms: list[str],
    term_profiles: dict[str, dict[str, float]],
) -> SearchResult | None:
    text = passage.text_normalized
    passage_terms = list(passage.content_lemmas) if passage.content_lemmas else list(passage.lemmas)

    exact_matches = max(
        _count_exact_query_matches(text, normalized_full_query),
        _count_exact_lemma_matches(passage_terms, query_lemmas),
        _count_exact_lemma_matches(passage_terms, effective_terms),
    )

    doc_length = max(1, len(passage_terms))

    matched_terms = 0
    total_term_hits = 0
    tfidf_score = 0.0
    positions_by_term: dict[str, list[int]] = {}

    for term, profile in term_profiles.items():
        positions = [idx for idx, lemma in enumerate(passage_terms) if lemma == term]
        hits = len(positions)

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

    score = tfidf_score * 120.0 * coverage
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
    if length_words > 220:
        score -= min(10.0, (length_words - 220) * 0.06)

    return SearchResult(
        passage=passage,
        score=score,
        exact_matches=exact_matches,
        matched_terms=matched_terms,
        total_term_hits=total_term_hits,
    )


def _count_exact_lemma_matches(passage_terms: list[str], query_terms: list[str]) -> int:
    if not passage_terms or not query_terms:
        return 0

    if len(query_terms) == 1:
        target = query_terms[0]
        return sum(1 for term in passage_terms if term == target)

    count = 0
    window = len(query_terms)
    for start in range(0, len(passage_terms) - window + 1):
        if passage_terms[start : start + window] == query_terms:
            count += 1
    return count


def _minimum_cover_span(positions_by_term: dict[str, list[int]]) -> int | None:
    matched_terms = {term: positions for term, positions in positions_by_term.items() if positions}
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


def _idf(index: PassageIndex, term: str) -> float:
    total_docs = max(1, len(index.passages))
    df = index.lemma_document_freq.get(term, 0)
    return log((total_docs + 1) / (df + 1)) + 1.0


def _count_exact_query_matches(text: str, normalized_query: str) -> int:
    """Cuenta coincidencias exactas de la consulta completa con limites de palabra."""
    if not normalized_query:
        return 0

    exact_pattern = re.compile(
        rf"(?<!\w){re.escape(normalized_query)}(?!\w)",
        flags=re.IGNORECASE,
    )
    return len(exact_pattern.findall(text))


# --- BÚSQUEDA SEMÁNTICA ---

def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calcula la similitud del coseno entre dos vectores."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    mag1 = sqrt(sum(a * a for a in v1))
    mag2 = sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)


def search_semantic_passages(
    index: PassageIndex,
    query: str,
    limit: int,
    chapter_filter: str | None = None,
    model: str = "nomic-embed-text"
) -> list[SearchResult]:
    """Busca pasajes usando embeddings y similitud del coseno."""
    normalized_query = normalize_text(query)
    if not normalized_query:
        raise ValueError("La consulta está vacía tras normalización.")

    # 1. Pedimos a Ollama el vector de la pregunta del usuario
    try:
        response = ollama.embeddings(model=model, prompt=normalized_query)
        query_embedding = response.get("embedding")
    except Exception as e:
        raise RuntimeError(f"Error al conectar con Ollama para la búsqueda semántica: {e}")

    if not query_embedding:
        raise RuntimeError("Ollama no devolvió un embedding válido para la consulta.")

    chapter_filter_normalized = normalize_text(chapter_filter) if chapter_filter else None
    results: list[SearchResult] = []

    # 2. Comparamos el vector de la pregunta con todos los pasajes del índice
    for passage in index.passages:
        if chapter_filter_normalized:
            chapter_normalized = normalize_text(passage.chapter or "")
            if chapter_filter_normalized not in chapter_normalized:
                continue
        
        # Ignoramos los pasajes que por algún motivo no tengan embedding
        if not passage.embedding:
            continue

        # 3. Calculamos la puntuación matemática de similitud
        sim_score = _cosine_similarity(query_embedding, passage.embedding)

        results.append(
            SearchResult(
                passage=passage,
                score=sim_score,
                exact_matches=0,  # En búsqueda semántica esto no aplica directamente
                matched_terms=0,  # Lo dejamos a 0 para mantener la compatibilidad
                total_term_hits=0,
            )
        )

    # 4. Ordenamos de mayor a menor similitud
    results.sort(key=lambda item: (-item.score, item.passage.order))
    return results[:limit]