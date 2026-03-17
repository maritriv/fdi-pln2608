"""Carga, limpieza y segmentación del corpus del Quijote."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.dammit import UnicodeDammit

from quijote_app.config import DEFAULT_SOURCE_CANDIDATES, MIN_WORDS_PER_PASSAGE
from quijote_app.models import CorpusMetadata, Passage
from quijote_app.utils import collapse_spaces, normalize_text

START_MARKER = "*** START OF THE PROJECT GUTENBERG"
END_MARKER = "*** END OF THE PROJECT GUTENBERG"

CHAPTER_RE = re.compile(r"^cap[ií]tulo\b", flags=re.IGNORECASE)
PART_RE = re.compile(r"^(primera|segunda)\s+parte\b", flags=re.IGNORECASE)
PROLOGUE_RE = re.compile(r"^pr[oó]logo\b", flags=re.IGNORECASE)

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
BLOCK_TAGS = ("div", "p", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6")


class CorpusError(RuntimeError):
    """Error de carga o parseo del corpus."""


def resolve_source_path(source: Path | None) -> Path:
    """Resuelve ruta de corpus a partir de opción explícita o candidatos por defecto."""
    if source is not None:
        resolved = source.expanduser()
        if not resolved.exists():
            raise CorpusError(f"La ruta del corpus no existe: {resolved}")
        return resolved

    for candidate in DEFAULT_SOURCE_CANDIDATES:
        if candidate.exists():
            return candidate

    candidates_str = ", ".join(str(p) for p in DEFAULT_SOURCE_CANDIDATES)
    raise CorpusError(
        "No se encontró corpus por defecto. "
        f"Busca en: {candidates_str}. Usa --source para indicar una ruta."
    )


def load_corpus(
    source: Path,
    min_words: int = MIN_WORDS_PER_PASSAGE,
) -> tuple[CorpusMetadata, list[Passage]]:
    """Carga el corpus, segmenta pasajes y devuelve metadatos."""
    source = source.expanduser().resolve()
    if not source.exists():
        raise CorpusError(f"La ruta del corpus no existe: {source}")

    html_text, selected_entry, source_kind = read_html_from_source(source)
    passages, chapters, parts = extract_passages_from_html(html_text, min_words=min_words)

    if not passages:
        raise CorpusError("El corpus no produjo pasajes útiles tras limpieza y segmentación.")

    metadata = CorpusMetadata(
        source_path=source,
        source_kind=source_kind,
        source_size=source.stat().st_size,
        source_mtime_ns=source.stat().st_mtime_ns,
        selected_entry=selected_entry,
        chapter_count=len(chapters),
        part_count=len(parts),
        passage_count=len(passages),
    )
    return metadata, passages


def read_html_from_source(source: Path) -> tuple[str, str | None, str]:
    """Lee HTML desde ZIP o desde fichero HTML directo."""
    suffix = source.suffix.lower()

    if suffix in {".htm", ".html"}:
        return _decode_bytes(source.read_bytes()), None, "html"

    if suffix != ".zip":
        raise CorpusError("Formato no soportado. Usa un .zip o .html/.htm")

    with zipfile.ZipFile(source, "r") as zf:
        html_entries = [entry for entry in zf.infolist() if entry.filename.lower().endswith((".htm", ".html"))]
        if not html_entries:
            raise CorpusError("El ZIP no contiene ficheros HTML.")

        main_entry = _select_main_html_entry(html_entries)
        data = zf.read(main_entry.filename)
        html_text = _decode_bytes(data)

    return html_text, main_entry.filename, "zip"


def extract_passages_from_html(
    html_text: str,
    min_words: int = MIN_WORDS_PER_PASSAGE,
) -> tuple[list[Passage], set[str], set[str]]:
    """Extrae pasajes recuperables preservando estructura de capítulos."""
    soup = BeautifulSoup(html_text, "html.parser")
    root = soup.body or soup

    markers_present = START_MARKER in html_text.upper() and END_MARKER in html_text.upper()
    collecting = not markers_present

    current_part: str | None = None
    current_chapter: str | None = None

    passages: list[Passage] = []
    chapter_labels: set[str] = set()
    part_labels: set[str] = set()
    order = 0

    for node in root.find_all(BLOCK_TAGS):
        text = collapse_spaces(node.get_text(" ", strip=True))
        if not text:
            continue

        upper_text = text.upper()
        if START_MARKER in upper_text:
            collecting = True
            continue
        if END_MARKER in upper_text:
            break
        if not collecting:
            continue

        if _is_noise_text(text):
            continue

        if node.name in HEADING_TAGS:
            if PART_RE.search(text):
                current_part = text
                part_labels.add(text)
                continue

            if CHAPTER_RE.search(text) or PROLOGUE_RE.search(text):
                current_chapter = text
                chapter_labels.add(_compose_chapter_label(current_part, current_chapter))
                continue

            if current_chapter is None and _is_prelim_heading(text):
                current_chapter = text
                chapter_labels.add(_compose_chapter_label(current_part, current_chapter))
            continue

        if _word_count(text) < min_words:
            continue

        normalized = normalize_text(text)
        if len(normalized) < 16:
            continue

        chapter_label = _compose_chapter_label(current_part, current_chapter)
        passages.append(
            Passage(
                passage_id=f"P{order:06d}",
                order=order,
                chapter=chapter_label,
                part=current_part,
                text_original=text,
                text_normalized=normalized,
            )
        )
        order += 1

    return passages, chapter_labels, part_labels


def _decode_bytes(data: bytes) -> str:
    """Decodifica bytes HTML con fallback robusto."""
    guessed = UnicodeDammit(data, ["utf-8", "utf-8-sig", "cp1252", "latin-1"]).unicode_markup
    if guessed is None:
        raise CorpusError("No se pudo decodificar el HTML del corpus.")
    return guessed


def _select_main_html_entry(entries: list[zipfile.ZipInfo]) -> zipfile.ZipInfo:
    """Selecciona el HTML principal si un ZIP tiene múltiples candidatos."""

    def score(entry: zipfile.ZipInfo) -> float:
        name = entry.filename.lower()
        value = float(entry.file_size)
        if "quijote" in name:
            value += 1_000_000
        if "2000-h" in name:
            value += 750_000
        if "index" in name:
            value -= 250_000
        return value

    return max(entries, key=score)


def _word_count(text: str) -> int:
    return len(text.split())


def _compose_chapter_label(part: str | None, chapter: str | None) -> str | None:
    if chapter and part and normalize_text(part) not in normalize_text(chapter):
        return f"{part} | {chapter}"
    return chapter or part


def _is_prelim_heading(text: str) -> bool:
    normalized = normalize_text(text)
    return normalized in {
        "tasa",
        "testimonio de las erratas",
        "el rey",
        "prologo",
        "al libro de don quijote de la mancha",
    }


def _is_noise_text(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return True

    noise_markers = (
        "project gutenberg",
        "this ebook",
        "produced by",
        "html version by",
        "updated editions",
    )
    if any(marker in normalized for marker in noise_markers):
        return True

    if normalized.count("capitulo") >= 3:
        return True

    if len(re.findall(r"\b[\wáéíóúüñ]+-\b", text, flags=re.IGNORECASE)) >= 3:
        return True

    if re.fullmatch(r"\d+", normalized):
        return True
    if re.fullmatch(r"[ivxlcdm]+", normalized):
        return True

    return False
