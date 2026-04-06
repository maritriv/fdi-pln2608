"""Utilidades NLP para lematizacion en espanol con preferencia por spaCy."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Any, Iterable

from quijote_app.models import Passage
from quijote_app.utils import normalize_text

try:  # pragma: no cover - depende del entorno local
    import spacy
except Exception:  # pragma: no cover - fallback defensivo
    spacy = None  # type: ignore[assignment]

SPACY_MODEL_NAME = "es_core_news_sm"
EXTRA_STOPWORDS = {
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


@lru_cache(maxsize=1)
def get_spanish_nlp() -> Any | None:
    """Carga spaCy en espanol; devuelve None si no esta disponible."""
    if spacy is None:
        return None

    try:
        return spacy.load(
            SPACY_MODEL_NAME,
            disable=[
                "ner",
                "parser",
                "senter",
            ],
        )
    except Exception:
        try:
            nlp = spacy.blank("es")
            if "lemmatizer" not in nlp.pipe_names:
                nlp.add_pipe("lemmatizer", config={"mode": "rule"})
            nlp.initialize()
            return nlp
        except Exception:
            return None


@lru_cache(maxsize=1)
def get_stopwords() -> set[str]:
    """Conjunto de stopwords normalizadas."""
    stopwords = set(EXTRA_STOPWORDS)
    nlp = get_spanish_nlp()
    if nlp is not None:
        stopwords.update(normalize_text(token) for token in nlp.Defaults.stop_words)
    return {token for token in stopwords if token}


def analyze_text(text: str) -> tuple[list[str], list[str]]:
    """Devuelve (lemmas, content_lemmas) de un texto."""
    if not text:
        return [], []

    nlp = get_spanish_nlp()
    if nlp is None:
        return _analyze_text_heuristic(text)

    doc = nlp(text)

    lemmas: list[str] = []
    content_lemmas: list[str] = []
    stopwords = get_stopwords()

    for token in doc:
        if token.is_space or token.is_punct:
            continue

        lemma = _select_lemma(token_text=token.text, lemma_candidate=token.lemma_)
        if not lemma or lemma.isdigit():
            continue

        lemmas.append(lemma)
        if not token.is_stop and lemma not in stopwords:
            content_lemmas.append(lemma)

    return lemmas, content_lemmas


def annotate_passage(passage: Passage) -> Passage:
    """Completa un pasaje con informacion lematizada."""
    lemmas, content_lemmas = analyze_text(passage.text_original)
    passage.lemmas = tuple(lemmas)
    passage.content_lemmas = tuple(content_lemmas)
    return passage


def annotate_passages(passages: Iterable[Passage]) -> list[Passage]:
    """Anota una coleccion de pasajes con lemas y contenido util."""
    return [annotate_passage(passage) for passage in passages]


def compute_document_frequencies(passages: Iterable[Passage]) -> dict[str, int]:
    """Calcula DF por lema usando terminos de contenido y fallback a lemas."""
    frequencies: Counter[str] = Counter()
    for passage in passages:
        terms = (
            set(passage.content_lemmas)
            if passage.content_lemmas
            else set(passage.lemmas)
        )
        for term in terms:
            frequencies[term] += 1
    return dict(frequencies)


def select_query_terms(query: str) -> tuple[list[str], list[str]]:
    """Lematiza consulta y retorna (lemmas, terminos_efectivos)."""
    lemmas, content_lemmas = analyze_text(query)
    effective_terms = content_lemmas if content_lemmas else lemmas
    return lemmas, effective_terms


def _analyze_text_heuristic(text: str) -> tuple[list[str], list[str]]:
    tokens = normalize_text(text).split()
    if not tokens:
        return [], []

    stopwords = get_stopwords()
    lemmas: list[str] = []
    content_lemmas: list[str] = []

    for token in tokens:
        lemma = _heuristic_lemma(token)
        if not lemma or lemma.isdigit():
            continue

        lemmas.append(lemma)
        if lemma not in stopwords:
            content_lemmas.append(lemma)

    return lemmas, content_lemmas


def _select_lemma(token_text: str, lemma_candidate: str | None) -> str:
    raw = (lemma_candidate or "").strip()
    if not raw or raw == "-PRON-":
        raw = token_text

    normalized = normalize_text(raw)
    if not normalized:
        return ""

    if " " in normalized:
        return normalized.split()[0]

    return normalized


def _heuristic_lemma(token: str) -> str:
    lemma = token

    # plural comun
    if lemma.endswith("ces") and len(lemma) > 4:
        lemma = lemma[:-3] + "z"
    elif lemma.endswith("es") and len(lemma) > 4 and lemma[-3] not in "aeiou":
        lemma = lemma[:-2]
    elif lemma.endswith("s") and len(lemma) > 3 and lemma[-2] in "aeiou":
        lemma = lemma[:-1]

    # verbos frecuentes en -ar
    for suffix in ("ando", "aba", "aban", "abas", "ado", "ada", "ados", "adas", "aron"):
        if lemma.endswith(suffix) and len(lemma) - len(suffix) >= 3:
            return lemma[: -len(suffix)] + "ar"

    # verbos frecuentes en -er/-ir
    for suffix in ("iendo", "ia", "ian", "ias", "ido", "ida", "idos", "idas", "ieron"):
        if lemma.endswith(suffix) and len(lemma) - len(suffix) >= 3:
            return lemma[: -len(suffix)] + "er"

    return lemma
