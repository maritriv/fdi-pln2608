"""Carga, limpieza y segmentacion del corpus del Quijote."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import re
import zipfile
from pathlib import Path

from quijote_app.config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_TARGET_WORDS,
    DEFAULT_SOURCE_CANDIDATES,
    MIN_WORDS_PER_BLOCK,
    MIN_WORDS_PER_PASSAGE,
)
from quijote_app.models import CorpusMetadata, Passage
from quijote_app.utils import collapse_spaces, normalize_text

MAX_WORDS_PER_UNIT = 90
HARD_MAX_WORDS_PER_UNIT = 140

START_MARKER = "*** START OF THE PROJECT GUTENBERG"
END_MARKER = "*** END OF THE PROJECT GUTENBERG"

CHAPTER_RE = re.compile(r"^cap[ii]tulo\b", flags=re.IGNORECASE)
PART_RE = re.compile(r"^(primera|segunda)\s+parte\b", flags=re.IGNORECASE)
PROLOGUE_RE = re.compile(r"^pr[oó]logo\b", flags=re.IGNORECASE)

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
BLOCK_TAGS = {"div", "p", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"}


class CorpusError(RuntimeError):
    """Error de carga o parseo del corpus."""


@dataclass
class _TextUnit:
    text: str
    chapter: str | None
    part: str | None


class _HtmlBlockExtractor(HTMLParser):
    """Extrae bloques de texto con su etiqueta origen sin usar librerias externas."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str]] = []
        self._stack: list[dict[str, object]] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        lowered = tag.lower()

        if lowered in {"script", "style"}:
            self._ignored_depth += 1
            return

        if self._ignored_depth > 0:
            return

        if lowered in BLOCK_TAGS:
            if self._stack:
                self._stack[-1]["has_block_child"] = True
            self._stack.append({"tag": lowered, "chunks": [], "has_block_child": False})
            return

        if lowered == "br" and self._stack:
            self._stack[-1]["chunks"].append(" ")

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        lowered = tag.lower()

        if lowered in {"script", "style"}:
            if self._ignored_depth > 0:
                self._ignored_depth -= 1
            return

        if self._ignored_depth > 0:
            return

        if lowered in BLOCK_TAGS:
            self._close_frame(lowered)

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._ignored_depth > 0:
            return
        if not self._stack:
            return
        self._stack[-1]["chunks"].append(data)

    def close(self) -> None:
        super().close()
        while self._stack:
            frame = self._stack.pop()
            self._emit_frame(frame)

    def _close_frame(self, tag: str) -> None:
        for idx in range(len(self._stack) - 1, -1, -1):
            if self._stack[idx]["tag"] == tag:
                while len(self._stack) - 1 >= idx:
                    frame = self._stack.pop()
                    self._emit_frame(frame)
                    if frame["tag"] == tag:
                        return
                return

    def _emit_frame(self, frame: dict[str, object]) -> None:
        tag = str(frame["tag"])
        chunks = frame["chunks"]
        text = collapse_spaces("".join(chunks))
        if not text:
            return

        has_block_child = bool(frame["has_block_child"])
        if tag in HEADING_TAGS or not has_block_child:
            self.blocks.append((tag, text))


def resolve_source_path(source: Path | None) -> Path:
    """Resuelve ruta de corpus a partir de opcion explicita o candidatos por defecto."""
    if source is not None:
        resolved = source.expanduser()
        if not resolved.exists():
            raise CorpusError(f"La ruta del corpus no existe: {resolved}")
        return resolved

    for candidate in DEFAULT_SOURCE_CANDIDATES:
        if candidate.exists():
            return candidate

    candidates_str = ", ".join(str(path) for path in DEFAULT_SOURCE_CANDIDATES)
    raise CorpusError(
        "No se encontro corpus por defecto. "
        f"Busca en: {candidates_str}. Usa --source para indicar una ruta."
    )


def load_corpus(
    source: Path,
    min_words: int = MIN_WORDS_PER_PASSAGE,
) -> tuple[CorpusMetadata, list[Passage]]:
    """Carga el corpus, segmenta en chunks con overlap y devuelve metadatos."""
    source = source.expanduser().resolve()
    if not source.exists():
        raise CorpusError(f"La ruta del corpus no existe: {source}")

    html_text, selected_entry, source_kind = read_html_from_source(source)
    passages, chapters, parts = extract_passages_from_html(
        html_text, min_words=min_words
    )

    if not passages:
        raise CorpusError(
            "El corpus no produjo pasajes utiles tras limpieza y segmentacion."
        )

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

    with zipfile.ZipFile(source, "r") as zipped:
        html_entries = [
            entry
            for entry in zipped.infolist()
            if entry.filename.lower().endswith((".htm", ".html"))
        ]
        if not html_entries:
            raise CorpusError("El ZIP no contiene ficheros HTML.")

        main_entry = _select_main_html_entry(html_entries)
        data = zipped.read(main_entry.filename)
        html_text = _decode_bytes(data)

    return html_text, main_entry.filename, "zip"


def extract_passages_from_html(
    html_text: str,
    min_words: int = MIN_WORDS_PER_PASSAGE,
) -> tuple[list[Passage], set[str], set[str]]:
    """Extrae pasajes en chunks con overlap preservando contexto de capitulo."""
    blocks = _extract_blocks_from_html(html_text)

    markers_present = (
        START_MARKER in html_text.upper() and END_MARKER in html_text.upper()
    )
    collecting = not markers_present

    current_part: str | None = None
    current_chapter: str | None = None

    units: list[_TextUnit] = []
    chapter_labels: set[str] = set()
    part_labels: set[str] = set()

    for tag, text in blocks:
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

        if tag in HEADING_TAGS:
            if PART_RE.search(text):
                current_part = text
                part_labels.add(text)
                continue

            if CHAPTER_RE.search(text) or PROLOGUE_RE.search(text):
                current_chapter = text
                chapter_labels.add(
                    _compose_chapter_label(current_part, current_chapter)
                )
                continue

            if current_chapter is None and _is_prelim_heading(text):
                current_chapter = text
                chapter_labels.add(
                    _compose_chapter_label(current_part, current_chapter)
                )
            continue

        if _word_count(text) < MIN_WORDS_PER_BLOCK:
            continue

        normalized = normalize_text(text)
        if len(normalized) < 16:
            continue

        chapter_label = _compose_chapter_label(current_part, current_chapter)

        for piece in _split_long_text_unit(text):
            units.append(
                _TextUnit(text=piece, chapter=chapter_label, part=current_part)
            )

    passages = _build_overlap_chunks(units=units, min_words=min_words)
    return passages, chapter_labels, part_labels


def _extract_blocks_from_html(html_text: str) -> list[tuple[str, str]]:
    parser = _HtmlBlockExtractor()
    parser.feed(html_text)
    parser.close()
    return parser.blocks


def _split_long_text_unit(text: str, max_words: int = MAX_WORDS_PER_UNIT) -> list[str]:
    """Divide un bloque largo en trozos mas pequenos, intentando respetar frases."""
    text = collapse_spaces(text)
    if _word_count(text) <= max_words:
        return [text]

    sentences = re.split(r"(?<=[.!?;:])\s+", text)
    sentences = [collapse_spaces(s) for s in sentences if collapse_spaces(s)]

    if not sentences:
        return _split_by_words(text, max_words=max_words)

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = _word_count(sentence)

        if sentence_words > HARD_MAX_WORDS_PER_UNIT:
            if current:
                chunks.append(collapse_spaces(" ".join(current)))
                current = []
                current_words = 0
            chunks.extend(_split_by_words(sentence, max_words=max_words))
            continue

        if current_words + sentence_words > max_words and current:
            chunks.append(collapse_spaces(" ".join(current)))
            current = [sentence]
            current_words = sentence_words
        else:
            current.append(sentence)
            current_words += sentence_words

    if current:
        chunks.append(collapse_spaces(" ".join(current)))

    return [
        chunk for chunk in chunks if chunk and _word_count(chunk) >= MIN_WORDS_PER_BLOCK
    ]


def _split_by_words(text: str, max_words: int = MAX_WORDS_PER_UNIT) -> list[str]:
    """Fallback bruto por ventanas de palabras."""
    words = text.split()
    chunks: list[str] = []

    for i in range(0, len(words), max_words):
        piece = " ".join(words[i : i + max_words])
        piece = collapse_spaces(piece)
        if piece:
            chunks.append(piece)

    return chunks


def _build_overlap_chunks(units: list[_TextUnit], min_words: int) -> list[Passage]:
    if not units:
        return []

    grouped_units: list[list[_TextUnit]] = []
    current_group: list[_TextUnit] = []
    current_key: tuple[str | None, str | None] | None = None

    for unit in units:
        key = (unit.part, unit.chapter)
        if current_key is None or key == current_key:
            current_group.append(unit)
            current_key = key
            continue

        grouped_units.append(current_group)
        current_group = [unit]
        current_key = key

    if current_group:
        grouped_units.append(current_group)

    passages: list[Passage] = []
    order = 0

    for group in grouped_units:
        word_counts = [_word_count(unit.text) for unit in group]
        start = 0

        while start < len(group):
            end = start
            chunk_words = 0

            while end < len(group) and chunk_words < CHUNK_TARGET_WORDS:
                chunk_words += word_counts[end]
                end += 1

            if end == start:
                end = start + 1
                chunk_words = word_counts[start]

            chunk_units = group[start:end]
            chunk_text = collapse_spaces(" ".join(unit.text for unit in chunk_units))

            for final_piece in _split_by_words(
                chunk_text, max_words=CHUNK_TARGET_WORDS + 20
            ):
                if _word_count(final_piece) >= min_words:
                    passages.append(
                        Passage(
                            passage_id=f"P{order:06d}",
                            order=order,
                            chapter=chunk_units[0].chapter,
                            part=chunk_units[0].part,
                            text_original=final_piece,
                            text_normalized=normalize_text(final_piece),
                        )
                    )
                    order += 1

            if end >= len(group):
                break

            overlap = 0
            next_start = end
            while next_start > start and overlap < CHUNK_OVERLAP_WORDS:
                next_start -= 1
                overlap += word_counts[next_start]

            if next_start <= start:
                next_start = start + 1

            start = next_start

    return passages


def _decode_bytes(data: bytes) -> str:
    """Decodifica bytes HTML con fallback robusto."""
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _select_main_html_entry(entries: list[zipfile.ZipInfo]) -> zipfile.ZipInfo:
    """Selecciona el HTML principal si un ZIP tiene multiples candidatos."""

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
    if (
        len(re.findall(r"\bde\b", normalized)) >= 8
        and normalized.count("capitulo") >= 1
    ):
        return True
    if len(re.findall(r"\bdonde\b", normalized)) >= 4:
        return True

    title_like_starts = ("de ", "donde ", "del ", "que trata ", "como ")
    fragments = re.split(r"(?<=[.:;])\s+|\s{2,}", normalized)
    fragments = [frag.strip() for frag in fragments if frag.strip()]
    if len(fragments) >= 4:
        title_like_count = sum(
            1 for frag in fragments if frag.startswith(title_like_starts)
        )
        if title_like_count >= 4:
            return True

    return False
