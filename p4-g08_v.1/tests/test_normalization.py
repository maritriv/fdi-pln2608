from quijote_app.utils import find_query_span, normalize_text, render_excerpt


def test_normalize_text_case_accents_and_punctuation() -> None:
    assert normalize_text("  DULCINEA,   del   Toboso!!! ") == "dulcinea del toboso"


def test_render_excerpt_highlights_accent_insensitive_match() -> None:
    text = "Y asi llamo a su dama Dulcinea del Toboso en su pensamiento."
    excerpt = render_excerpt(text, "dulcinea del toboso", max_chars=200)
    assert "[Dulcinea del Toboso]" in excerpt


def test_find_query_span_returns_none_when_no_match() -> None:
    assert find_query_span("En un lugar de la Mancha", "Rocinante") is None
