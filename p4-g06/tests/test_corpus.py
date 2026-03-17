import zipfile

from quijote_app.corpus import extract_passages_from_html, read_html_from_source


def test_extract_passages_from_html_detects_chapter_and_filters_footer() -> None:
    html = """
    <html><body>
      <div>*** START OF THE PROJECT GUTENBERG EBOOK DON QUIJOTE ***</div>
      <h3>Capítulo I. Que trata de la condición y ejercicio del famoso hidalgo</h3>
      <p>En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo.</p>
      <p>Tenía en su casa una ama que pasaba de los cuarenta, y una sobrina que no llegaba a los veinte.</p>
      <div>*** END OF THE PROJECT GUTENBERG EBOOK DON QUIJOTE ***</div>
      <p>Project Gutenberg License footer text.</p>
    </body></html>
    """
    passages, chapters, _parts = extract_passages_from_html(html, min_words=8)
    assert len(passages) == 2
    assert any("Capítulo I" in chapter for chapter in chapters)
    assert all("project gutenberg" not in p.text_normalized for p in passages)


def test_read_html_from_zip_selects_main_entry(tmp_path) -> None:
    zip_path = tmp_path / "quijote.zip"
    main_html = "<html><body><p>" + ("palabra " * 200) + "</p></body></html>"
    small_html = "<html><body><p>índice</p></body></html>"

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("index.html", small_html)
        zf.writestr("2000-h.htm", main_html)

    html_text, selected_entry, source_kind = read_html_from_source(zip_path)
    assert source_kind == "zip"
    assert selected_entry == "2000-h.htm"
    assert "palabra palabra" in html_text

