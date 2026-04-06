from pathlib import Path

from quijote_app.models import CorpusMetadata, Passage, PassageIndex
from quijote_app.nlp import annotate_passages, compute_document_frequencies
from quijote_app.search import search_passages
from quijote_app.utils import normalize_text


def _build_index(passages: list[Passage]) -> PassageIndex:
    annotated_passages = annotate_passages(passages)
    metadata = CorpusMetadata(
        source_path=Path("dummy.html"),
        source_kind="html",
        source_size=0,
        source_mtime_ns=0,
        selected_entry=None,
        chapter_count=len({p.chapter for p in passages if p.chapter}),
        part_count=len({p.part for p in passages if p.part}),
        passage_count=len(passages),
    )
    return PassageIndex(
        metadata=metadata,
        built_at_iso="2026-01-01T00:00:00+00:00",
        pipeline_version="1.0",
        passages=annotated_passages,
        lemma_document_freq=compute_document_frequencies(annotated_passages),
    )


def test_search_simple_term() -> None:
    passages = [
        Passage(
            passage_id="P000000",
            order=0,
            chapter="Capitulo I",
            part="Primera parte",
            text_original="En un lugar de la Mancha, de cuyo nombre no quiero acordarme.",
            text_normalized=normalize_text(
                "En un lugar de la Mancha, de cuyo nombre no quiero acordarme."
            ),
        ),
        Passage(
            passage_id="P000001",
            order=1,
            chapter="Capitulo XIII",
            part="Primera parte",
            text_original="He querido llamar a la mia Dulcinea del Toboso.",
            text_normalized=normalize_text(
                "He querido llamar a la mia Dulcinea del Toboso."
            ),
        ),
        Passage(
            passage_id="P000002",
            order=2,
            chapter="Capitulo XXV",
            part="Primera parte",
            text_original="La sin par Dulcinea del Toboso era su dama.",
            text_normalized=normalize_text(
                "La sin par Dulcinea del Toboso era su dama."
            ),
        ),
    ]

    index = _build_index(passages)
    results = search_passages(index=index, query="dulcinea", limit=5)

    assert len(results) == 2
    assert all("dulcinea" in result.passage.text_normalized for result in results)


def test_search_matches_singular_and_plural_by_lemma() -> None:
    passages = [
        Passage(
            passage_id="P100000",
            order=0,
            chapter="Capitulo X",
            part="Primera parte",
            text_original="Aparecio un molino en la llanura.",
            text_normalized=normalize_text("Aparecio un molino en la llanura."),
        ),
        Passage(
            passage_id="P100001",
            order=1,
            chapter="Capitulo XI",
            part="Primera parte",
            text_original="Vio despues muchos molinos en el camino.",
            text_normalized=normalize_text("Vio despues muchos molinos en el camino."),
        ),
        Passage(
            passage_id="P100002",
            order=2,
            chapter="Capitulo XII",
            part="Primera parte",
            text_original="Un remolino levanto polvo al atardecer.",
            text_normalized=normalize_text("Un remolino levanto polvo al atardecer."),
        ),
    ]

    index = _build_index(passages)
    results_singular = search_passages(index=index, query="molino", limit=10)
    results_plural = search_passages(index=index, query="molinos", limit=10)

    ids_singular = {result.passage.passage_id for result in results_singular}
    ids_plural = {result.passage.passage_id for result in results_plural}

    assert ids_singular == {"P100000", "P100001"}
    assert ids_plural == {"P100000", "P100001"}


def test_search_matches_basic_verb_variants_by_lemma() -> None:
    passages = [
        Passage(
            passage_id="PV0001",
            order=0,
            chapter="Capitulo I",
            part="Primera parte",
            text_original="Yo quiero cantar ahora.",
            text_normalized=normalize_text("Yo quiero cantar ahora."),
        ),
        Passage(
            passage_id="PV0002",
            order=1,
            chapter="Capitulo II",
            part="Primera parte",
            text_original="El caballero cantaba muy fuerte.",
            text_normalized=normalize_text("El caballero cantaba muy fuerte."),
        ),
        Passage(
            passage_id="PV0003",
            order=2,
            chapter="Capitulo III",
            part="Primera parte",
            text_original="Iban cantando en el camino.",
            text_normalized=normalize_text("Iban cantando en el camino."),
        ),
    ]

    index = _build_index(passages)
    results = search_passages(index=index, query="cantar", limit=10)
    ids = {result.passage.passage_id for result in results}

    assert ids == {"PV0001", "PV0002", "PV0003"}


def test_search_ignores_stopwords_when_query_has_content_terms() -> None:
    passages = [
        Passage(
            passage_id="PS0001",
            order=0,
            chapter="Capitulo S1",
            part="Primera parte",
            text_original="El de la y en por para.",
            text_normalized=normalize_text("El de la y en por para."),
        ),
        Passage(
            passage_id="PS0002",
            order=1,
            chapter="Capitulo S2",
            part="Primera parte",
            text_original="Dulcinea sonaba en su memoria.",
            text_normalized=normalize_text("Dulcinea sonaba en su memoria."),
        ),
    ]

    index = _build_index(passages)
    results = search_passages(index=index, query="el de la dulcinea", limit=10)

    assert len(results) == 1
    assert results[0].passage.passage_id == "PS0002"


def test_search_with_chapter_filter() -> None:
    passages = [
        Passage(
            passage_id="PF0001",
            order=0,
            chapter="Capitulo XIII",
            part="Primera parte",
            text_original="Dulcinea reaparece en este capitulo.",
            text_normalized=normalize_text("Dulcinea reaparece en este capitulo."),
        ),
        Passage(
            passage_id="PF0002",
            order=1,
            chapter="Capitulo XXV",
            part="Primera parte",
            text_original="Dulcinea vuelve a mencionarse en este otro.",
            text_normalized=normalize_text(
                "Dulcinea vuelve a mencionarse en este otro."
            ),
        ),
    ]

    index = _build_index(passages)
    results = search_passages(
        index=index, query="dulcinea", limit=10, chapter_filter="xxv"
    )

    assert len(results) == 1
    assert results[0].passage.passage_id == "PF0002"


def test_search_tfidf_prioritizes_term_coverage() -> None:
    passages = [
        Passage(
            passage_id="PT0001",
            order=0,
            chapter="Capitulo T1",
            part="Primera parte",
            text_original="Insula y viento aparecen juntos.",
            text_normalized=normalize_text("Insula y viento aparecen juntos."),
        ),
        Passage(
            passage_id="PT0002",
            order=1,
            chapter="Capitulo T2",
            part="Primera parte",
            text_original="Insula insula insula insula.",
            text_normalized=normalize_text("Insula insula insula insula."),
        ),
    ]

    index = _build_index(passages)
    results = search_passages(index=index, query="las insulas de los vientos", limit=10)

    assert results[0].passage.passage_id == "PT0001"
