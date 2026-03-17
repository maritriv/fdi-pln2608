from quijote_app.utils import expand_term_variants, find_query_span, normalize_text, render_excerpt


def test_normalize_text_case_accents_and_punctuation() -> None:
    assert normalize_text("  DULCÍNEA,   del   Toboso!!! ") == "dulcinea del toboso"


def test_render_excerpt_highlights_accent_insensitive_match() -> None:
    text = "Y así llamó a su dama Dulcínea del Toboso en su pensamiento."
    excerpt = render_excerpt(text, "dulcinea del toboso", max_chars=200)
    assert "[Dulcínea del Toboso]" in excerpt


def test_find_query_span_returns_none_when_no_match() -> None:
    assert find_query_span("En un lugar de la Mancha", "Rocinante") is None



def test_expand_term_variants_supports_singular_plural() -> None:
    assert "molinos" in expand_term_variants("molino")
    assert "molino" in expand_term_variants("molinos")


def test_render_excerpt_highlights_plural_for_singular_query() -> None:
    text = "Vio muchos molinos en el campo."
    excerpt = render_excerpt(text, "molino", max_chars=200)
    assert "[molinos]" in excerpt.lower()


def test_render_excerpt_highlights_singular_for_plural_query() -> None:
    text = "Vio un molino en el campo."
    excerpt = render_excerpt(text, "molinos", max_chars=200)
    assert "[molino]" in excerpt.lower()
