"""Preparacion reproducible del corpus NER de la preentrega."""

from __future__ import annotations

import json
from pathlib import Path


PREENTREGA_TO_BIO = {
    "o": "O",
    "pi": "B-PER",
    "pc": "I-PER",
    "li": "B-LOC",
    "lc": "I-LOC",
}


def _normalise_label(label: str) -> str:
    key = label.strip().lower()
    if key not in PREENTREGA_TO_BIO:
        raise ValueError(f"Etiqueta NER desconocida: {label!r}")
    return PREENTREGA_TO_BIO[key]


def load_merged_json(input_path: str | Path) -> list[dict]:
    """Carga el JSON fusionado de la preentrega."""

    path = Path(input_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("El corpus fusionado debe ser una lista de frases.")
    return data


def convert_merged_to_conll(
    input_path: str | Path,
    output_path: str | Path,
) -> dict:
    """Convierte `merged.json` a CoNLL BIO y devuelve estadisticas basicas.

    El corpus fusionado tokeniza espacios y saltos de linea como tokens propios.
    CoNLL usa una linea por token no vacio, por lo que se omite cualquier token
    puramente blanco y se conservan las frases con una linea en blanco.
    """

    data = load_merged_json(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    label_counts = {label: 0 for label in PREENTREGA_TO_BIO.values()}
    total_tokens = 0
    written_tokens = 0
    skipped_whitespace = 0

    for index, item in enumerate(data):
        tokens = item.get("tokens")
        labels = item.get("labels")
        if not isinstance(tokens, list) or not isinstance(labels, list):
            raise ValueError(f"Frase {index}: faltan listas tokens/labels.")
        if len(tokens) != len(labels):
            raise ValueError(
                f"Frase {index}: tokens ({len(tokens)}) y labels "
                f"({len(labels)}) no tienen la misma longitud."
            )

        phrase_written = False
        for token, label in zip(tokens, labels):
            if not isinstance(token, str):
                raise ValueError(f"Frase {index}: token no textual {token!r}.")

            total_tokens += 1
            bio_label = _normalise_label(str(label))
            label_counts[bio_label] += 1

            if token.strip() == "":
                skipped_whitespace += 1
                continue

            safe_token = token.replace("\n", "\\n").replace("\r", "\\r")
            lines.append(f"{safe_token} {bio_label}")
            written_tokens += 1
            phrase_written = True

        if phrase_written:
            lines.append("")

    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    entity_tokens = sum(count for label, count in label_counts.items() if label != "O")
    return {
        "sentences": len(data),
        "source_tokens": total_tokens,
        "written_tokens": written_tokens,
        "skipped_whitespace_tokens": skipped_whitespace,
        "entity_tokens": entity_tokens,
        "label_counts": label_counts,
        "input": str(input_path),
        "output": str(output_path),
    }


__all__ = ["PREENTREGA_TO_BIO", "convert_merged_to_conll", "load_merged_json"]
