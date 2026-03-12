#!/usr/bin/env -S uv run
# /// script
# dependencies = ["typer"]
# ///

import sys
from pathlib import Path
import typer
from collections import Counter

app = typer.Typer(help="PLNCG26 encoder/decoder")

# ─────────────────────────────────────────────
# CONSTANTES DEL FORMATO PLNCG26
# ─────────────────────────────────────────────

MAGIC = 20
SHIFT = ord("a") - MAGIC

SPACE = 0x0B
NEWLINE = 0x0A

CAPITALIZE = 0x35
ACUTE = 0x32
UMLAUT = 0x33
TILDE = 0x34

DIGIT_BASE = 0x60

PUNCT = {
    ".": 0x46,
    ",": 0x47,
    ";": 0x48,
    ":": 0x49,
    "!": 0x4A,
    "?": 0x4B,
    "(": 0x4C,
    ")": 0x4D,
    "-": 0x4E,
    "—": 0x4E,
    '"': 0x4F,
    "«": 0x50,
    "»": 0x50,
}

PUNCT_INV = {v: k for k, v in PUNCT.items()}

ACCENTS = {
    "á": ("a", ACUTE),
    "é": ("e", ACUTE),
    "í": ("i", ACUTE),
    "ó": ("o", ACUTE),
    "ú": ("u", ACUTE),
    "Á": ("A", ACUTE),
    "É": ("E", ACUTE),
    "Í": ("I", ACUTE),
    "Ó": ("O", ACUTE),
    "Ú": ("U", ACUTE),
    "ü": ("u", UMLAUT),
    "Ü": ("U", UMLAUT),
    "ñ": ("n", TILDE),
    "Ñ": ("N", TILDE),
}

ACUTE_MAP = {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú"}
UMLAUT_MAP = {"u": "ü"}
TILDE_MAP = {"n": "ñ"}


# ─────────────────────────────────────────────
# ENCODE
# ─────────────────────────────────────────────

def encode_letter(ch: str) -> bytes:
    upper = ch.isupper()
    base = ord(ch.lower()) - SHIFT
    if upper:
        return bytes([base, CAPITALIZE])
    return bytes([base])


def encode_text(text: str) -> bytes:

    out = bytearray()

    for ch in text:

        if ch == " ":
            out.append(SPACE)

        elif ch == "\n":
            out.append(NEWLINE)

        elif ch in ACCENTS:
            base, mod = ACCENTS[ch]
            out += encode_letter(base)
            out.append(mod)

        elif ch.lower() in "abcdefghijklmnopqrstuvwxyz":
            out += encode_letter(ch)

        elif ch.isdigit():
            out.append(DIGIT_BASE + int(ch))

        elif ch in PUNCT:
            out.append(PUNCT[ch])

    return bytes(out)


# ─────────────────────────────────────────────
# DECODE
# ─────────────────────────────────────────────

def decode_bytes(data: bytes) -> str:

    result = []
    quote_open = True

    for b in data:

        if b == SPACE:
            result.append(" ")

        elif b == NEWLINE:
            result.append("\n")

        elif b == CAPITALIZE and result:
            result[-1] = result[-1].upper()

        elif b == ACUTE and result:
            result[-1] = ACUTE_MAP.get(result[-1], result[-1])

        elif b == UMLAUT and result:
            result[-1] = UMLAUT_MAP.get(result[-1], result[-1])

        elif b == TILDE and result:
            result[-1] = TILDE_MAP.get(result[-1], result[-1])

        elif DIGIT_BASE <= b <= DIGIT_BASE + 9:
            result.append(str(b - DIGIT_BASE))

        elif b == 0x50:
            result.append("«" if quote_open else "»")
            quote_open = not quote_open

        elif b in PUNCT_INV:
            result.append(PUNCT_INV[b])

        elif MAGIC <= b <= MAGIC + 25:
            result.append(chr(b + SHIFT))

        else:
            result.append(f"[{b}]")

    return "".join(result)


# ─────────────────────────────────────────────
# DETECT
# ─────────────────────────────────────────────

def detect_plncg26(data: bytes) -> float:

    freq = Counter(data)
    total = len(data)

    if total == 0:
        return 0.0

    valid = 0

    for b, count in freq.items():

        if b in (
            SPACE,
            NEWLINE,
            CAPITALIZE,
            ACUTE,
            UMLAUT,
            TILDE,
        ):
            valid += count

        elif MAGIC <= b <= MAGIC + 25:
            valid += count

        elif DIGIT_BASE <= b <= DIGIT_BASE + 9:
            valid += count

        elif b in PUNCT_INV:
            valid += count

    return valid / total


# ─────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────

@app.command()
def encode(file: Path):
    """
    Convierte UTF8 → PLNCG26
    """

    text = file.read_text(encoding="utf-8")
    data = encode_text(text)

    sys.stdout.buffer.write(data)


@app.command()
def decode(file: Path):
    """
    Convierte PLNCG26 → UTF8
    """

    data = file.read_bytes()
    text = decode_bytes(data)

    sys.stdout.write(text)


@app.command()
def detect(file: Path):
    """
    Calcula probabilidad de que un fichero sea PLNCG26
    """

    data = file.read_bytes()
    p = detect_plncg26(data)

    print(f"Probabilidad PLNCG26: {p:.2%}")


if __name__ == "__main__":
    app()
