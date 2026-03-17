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

