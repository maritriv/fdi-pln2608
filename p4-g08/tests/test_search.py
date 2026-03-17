from pathlib import Path

from quijote_app.models import CorpusMetadata, Passage, PassageIndex
from quijote_app.search import search_passages
from quijote_app.utils import normalize_text


def _build_dummy_index() -> PassageIndex:
    passages = [
        Passage(
            passage_id="P000000",
            order=0,
            chapter="Capítulo I",
            part="Primera parte",
            text_original="En un lugar de la Mancha, de cuyo nombre no quiero acordarme...",
            text_normalized=normalize_text("En un lugar de la Mancha, de cuyo nombre no quiero acordarme..."),
        ),
        Passage(
            passage_id="P000001",
            order=1,
            chapter="Capítulo XIII",
            part="Primera parte",
            text_original="... he querido llamar a la mía Dulcinea del Toboso ...",
            text_normalized=normalize_text("... he querido llamar a la mía Dulcinea del Toboso ..."),
        ),
        Passage(
            passage_id="P000002",
            order=2,
            chapter="Capítulo XXV",
            part="Primera parte",
            text_original="... la sin par Dulcinea del Toboso ...",
            text_normalized=normalize_text("... la sin par Dulcinea del Toboso ..."),
        ),
    ]

    metadata = CorpusMetadata(
        source_path=Path("dummy.html"),
        source_kind="html",
        source_size=0,
        source_mtime_ns=0,
        selected_entry=None,
        chapter_count=3,
        part_count=1,
        passage_count=len(passages),
    )
    return PassageIndex(
        metadata=metadata,
        built_at_iso="2026-01-01T00:00:00+00:00",
        pipeline_version="1.0.0",
        passages=passages,
    )


def test_search_simple_term() -> None:
    index = _build_dummy_index()
    results = search_passages(index=index, query="dulcinea", limit=5)
    assert len(results) == 2
    assert all("dulcinea" in r.passage.text_normalized for r in results)


def test_search_phrase_prioritizes_exact_match() -> None:
    index = _build_dummy_index()
    results = search_passages(index=index, query="dulcinea del toboso", limit=5)
    assert len(results) == 2
    assert results[0].exact_matches >= 1
    assert "dulcinea del toboso" in results[0].passage.text_normalized


def test_search_with_chapter_filter() -> None:
    index = _build_dummy_index()
    results = search_passages(index=index, query="dulcinea", limit=5, chapter_filter="xxv")
    assert len(results) == 1
    assert results[0].passage.chapter == "Capítulo XXV"



def test_search_matches_singular_and_plural_forms() -> None:
    passages = [
        Passage(
            passage_id="P100000",
            order=0,
            chapter="Capitulo X",
            part="Primera parte",
            text_original="... molino ...",
            text_normalized=normalize_text("... molino ..."),
        ),
        Passage(
            passage_id="P100001",
            order=1,
            chapter="Capitulo XI",
            part="Primera parte",
            text_original="... molinos ...",
            text_normalized=normalize_text("... molinos ..."),
        ),
        Passage(
            passage_id="P100002",
            order=2,
            chapter="Capitulo XII",
            part="Primera parte",
            text_original="... remolino ...",
            text_normalized=normalize_text("... remolino ..."),
        ),
    ]

    metadata = CorpusMetadata(
        source_path=Path("dummy.html"),
        source_kind="html",
        source_size=0,
        source_mtime_ns=0,
        selected_entry=None,
        chapter_count=3,
        part_count=1,
        passage_count=len(passages),
    )
    index = PassageIndex(
        metadata=metadata,
        built_at_iso="2026-01-01T00:00:00+00:00",
        pipeline_version="1.0.0",
        passages=passages,
    )

    results_singular = search_passages(index=index, query="molino", limit=10)
    results_plural = search_passages(index=index, query="molinos", limit=10)

    assert len(results_singular) == 2
    assert len(results_plural) == 2
    assert all("remolino" not in result.passage.text_normalized for result in results_singular)
    assert all("remolino" not in result.passage.text_normalized for result in results_plural)


def test_search_matches_basic_verb_variants_by_root() -> None:
    passages = [
        Passage(
            passage_id="PV0001",
            order=0,
            chapter="Capitulo I",
            part="Primera parte",
            text_original="yo quiero cantar ahora",
            text_normalized=normalize_text("yo quiero cantar ahora"),
        ),
        Passage(
            passage_id="PV0002",
            order=1,
            chapter="Capitulo II",
            part="Primera parte",
            text_original="el caballero cantaba muy fuerte",
            text_normalized=normalize_text("el caballero cantaba muy fuerte"),
        ),
        Passage(
            passage_id="PV0003",
            order=2,
            chapter="Capitulo III",
            part="Primera parte",
            text_original="iban cantando en el camino",
            text_normalized=normalize_text("iban cantando en el camino"),
        ),
        Passage(
            passage_id="PV0004",
            order=3,
            chapter="Capitulo IV",
            part="Primera parte",
            text_original="esto era puro encanto",
            text_normalized=normalize_text("esto era puro encanto"),
        ),
    ]

    metadata = CorpusMetadata(
        source_path=Path("dummy.html"),
        source_kind="html",
        source_size=0,
        source_mtime_ns=0,
        selected_entry=None,
        chapter_count=4,
        part_count=1,
        passage_count=len(passages),
    )
    index = PassageIndex(
        metadata=metadata,
        built_at_iso="2026-01-01T00:00:00+00:00",
        pipeline_version="1.0.0",
        passages=passages,
    )

    results = search_passages(index=index, query="cantar", limit=10)
    ids = {result.passage.passage_id for result in results}

    assert "PV0001" in ids
    assert "PV0002" in ids
    assert "PV0003" in ids
    assert "PV0004" not in ids


def test_search_ignores_stopword_noise_when_query_has_content_terms() -> None:
    index = _build_dummy_index()
    results = search_passages(index=index, query="el dulcinea", limit=5)
    assert len(results) == 2
    assert all("dulcinea" in result.passage.text_normalized for result in results)


def test_search_multiterm_prioritizes_content_terms_and_variants() -> None:
    passages = [
        Passage(
            passage_id="PM0001",
            order=0,
            chapter="Capitulo A",
            part="Primera parte",
            text_original="las de los la de los",
            text_normalized=normalize_text("las de los la de los"),
        ),
        Passage(
            passage_id="PM0002",
            order=1,
            chapter="Capitulo B",
            part="Primera parte",
            text_original="hablo de insulas lejanas",
            text_normalized=normalize_text("hablo de insulas lejanas"),
        ),
        Passage(
            passage_id="PM0003",
            order=2,
            chapter="Capitulo C",
            part="Primera parte",
            text_original="soplaban fuertes vientos del norte",
            text_normalized=normalize_text("soplaban fuertes vientos del norte"),
        ),
        Passage(
            passage_id="PM0004",
            order=3,
            chapter="Capitulo D",
            part="Primera parte",
            text_original="las insulas estaban cerca del viento marino",
            text_normalized=normalize_text("las insulas estaban cerca del viento marino"),
        ),
    ]

    metadata = CorpusMetadata(
        source_path=Path("dummy.html"),
        source_kind="html",
        source_size=0,
        source_mtime_ns=0,
        selected_entry=None,
        chapter_count=4,
        part_count=1,
        passage_count=len(passages),
    )
    index = PassageIndex(
        metadata=metadata,
        built_at_iso="2026-01-01T00:00:00+00:00",
        pipeline_version="1.0.0",
        passages=passages,
    )

    results = search_passages(index=index, query="las insulas de los vientos", limit=10)
    ids = [result.passage.passage_id for result in results]

    assert "PM0004" in ids
    assert ids[0] == "PM0004"
    assert "PM0001" not in ids


def test_search_only_stopwords_falls_back_to_stopword_tokens() -> None:
    passages = [
        Passage(
            passage_id="PS0001",
            order=0,
            chapter="Capitulo S1",
            part="Primera parte",
            text_original="el de la",
            text_normalized=normalize_text("el de la"),
        ),
        Passage(
            passage_id="PS0002",
            order=1,
            chapter="Capitulo S2",
            part="Primera parte",
            text_original="viento fuerte",
            text_normalized=normalize_text("viento fuerte"),
        ),
    ]

    metadata = CorpusMetadata(
        source_path=Path("dummy.html"),
        source_kind="html",
        source_size=0,
        source_mtime_ns=0,
        selected_entry=None,
        chapter_count=2,
        part_count=1,
        passage_count=len(passages),
    )
    index = PassageIndex(
        metadata=metadata,
        built_at_iso="2026-01-01T00:00:00+00:00",
        pipeline_version="1.0.0",
        passages=passages,
    )

    results = search_passages(index=index, query="el de la", limit=10)
    ids = {result.passage.passage_id for result in results}

    assert "PS0001" in ids
    assert "PS0002" not in ids


def test_search_tfidf_prioritizes_coverage_over_single_term_repetition() -> None:
    passages = [
        Passage(
            passage_id="PT0001",
            order=0,
            chapter="Capitulo T1",
            part="Primera parte",
            text_original="insula viento",
            text_normalized=normalize_text("insula viento"),
        ),
        Passage(
            passage_id="PT0002",
            order=1,
            chapter="Capitulo T2",
            part="Primera parte",
            text_original="insula insula insula insula",
            text_normalized=normalize_text("insula insula insula insula"),
        ),
    ]

    metadata = CorpusMetadata(
        source_path=Path("dummy.html"),
        source_kind="html",
        source_size=0,
        source_mtime_ns=0,
        selected_entry=None,
        chapter_count=2,
        part_count=1,
        passage_count=len(passages),
    )
    index = PassageIndex(
        metadata=metadata,
        built_at_iso="2026-01-01T00:00:00+00:00",
        pipeline_version="1.0.0",
        passages=passages,
    )

    results = search_passages(index=index, query="las insulas de los vientos", limit=10)

    assert results[0].passage.passage_id == "PT0001"
