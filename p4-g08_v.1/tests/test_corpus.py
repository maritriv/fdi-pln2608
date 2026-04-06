import zipfile

from quijote_app.corpus import extract_passages_from_html, read_html_from_source


def test_extract_passages_from_html_detects_chapter_and_filters_footer() -> None:
    html = """
    <html><body>
      <div>*** START OF THE PROJECT GUTENBERG EBOOK DON QUIJOTE ***</div>
      <h3>Capitulo I. Que trata de la condicion y ejercicio del famoso hidalgo</h3>
      <p>En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivia un hidalgo.</p>
      <p>Tenia en su casa una ama que pasaba de los cuarenta, y una sobrina que no llegaba a los veinte.</p>
      <div>*** END OF THE PROJECT GUTENBERG EBOOK DON QUIJOTE ***</div>
      <p>Project Gutenberg License footer text.</p>
    </body></html>
    """
    passages, chapters, _parts = extract_passages_from_html(html, min_words=8)

    assert len(passages) >= 1
    assert any("capitulo i" in chapter.lower() for chapter in chapters)
    assert all("project gutenberg" not in passage.text_normalized for passage in passages)


def test_read_html_from_zip_selects_main_entry(tmp_path) -> None:
    zip_path = tmp_path / "quijote.zip"
    main_html = "<html><body><p>" + ("palabra " * 200) + "</p></body></html>"
    small_html = "<html><body><p>indice</p></body></html>"

    with zipfile.ZipFile(zip_path, "w") as zipped:
        zipped.writestr("index.html", small_html)
        zipped.writestr("2000-h.htm", main_html)

    html_text, selected_entry, source_kind = read_html_from_source(zip_path)
    assert source_kind == "zip"
    assert selected_entry == "2000-h.htm"
    assert "palabra palabra" in html_text


def test_extract_passages_builds_overlap_chunks() -> None:
    paragraph1 = " ".join(["palabra1"] * 30)
    paragraph2 = " ".join(["palabra2"] * 30)
    paragraph3 = " ".join(["palabra3"] * 30)
    paragraph4 = "frase compartida " + " ".join(["palabra4"] * 28)
    paragraph5 = " ".join(["palabra5"] * 30)
    paragraph6 = " ".join(["palabra6"] * 30)

    html = f"""
    <html><body>
      <h3>Capitulo XV</h3>
      <p>{paragraph1}</p>
      <p>{paragraph2}</p>
      <p>{paragraph3}</p>
      <p>{paragraph4}</p>
      <p>{paragraph5}</p>
      <p>{paragraph6}</p>
    </body></html>
    """

    passages, _chapters, _parts = extract_passages_from_html(html, min_words=20)

    assert len(passages) >= 2
    overlap_hits = ["frase compartida" in passage.text_normalized for passage in passages]
    assert sum(overlap_hits) >= 2
