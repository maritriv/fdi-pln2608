#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["typer"]
# ///

"""Conversor entre PLNCG26 y UTF-8 para la práctica P3 (Detectives de criptoglifos).

Este script implementa una conversión bidireccional entre:

- texto codificado en el plano binario `PLNCG26`
- texto Unicode estándar (`UTF-8`)

Comandos disponibles
--------------------
- `decode`: convierte un fichero PLNCG26 a texto UTF-8 legible.
- `encode`: convierte un fichero UTF-8 a su representación PLNCG26.
- `detect`: estima de forma heurística la probabilidad de que un fichero
  pertenezca al plano PLNCG26.

Uso
---
    uv run fdi-pln-2608-p3.py decode <FICHERO_BIN>
    uv run fdi-pln-2608-p3.py encode <FICHERO_UTF8>
    uv run fdi-pln-2608-p3.py detect <FICHERO>

Ejemplos
--------
1. Decodificar PLNCG26 -> UTF-8:
   uv run fdi-pln-2608-p3.py decode principal.bin > salida.txt

2. Codificar UTF-8 -> PLNCG26:
   uv run fdi-pln-2608-p3.py encode texto.txt > salida.bin

3. Detección opcional:
   uv run fdi-pln-2608-p3.py detect principal.bin

Descripción general del sistema
-------------------------------
PLNCG26 no representa necesariamente cada carácter con un único byte.

El sistema se organiza en tres tipos de elementos:

1. Letras base:
   el rango de bytes `20..45` representa las letras `a..z`.

2. Símbolos de un solo byte:
   espacio, salto de línea, dígitos y varios signos de puntuación tienen una
   representación directa de un byte.

3. Modificadores:
   algunos bytes no representan un carácter independiente, sino que modifican
   la letra base inmediatamente anterior. Se usan para:
   - mayúsculas
   - acento agudo
   - diéresis
   - tilde

Ejemplos conceptuales:
- `a`  -> byte base de `a`
- `A`  -> byte base de `a` + modificador de mayúscula
- `é`  -> byte base de `e` + modificador de acento
- `Ñ`  -> byte base de `n` + modificador de mayúscula + modificador de tilde

Cómo se dedujo la tabla
-----------------------
La tabla de correspondencias se reconstruyó mediante análisis de los ficheros
proporcionados en la práctica:

- `10` y `11` se identificaron por frecuencia y posición como salto de línea y
  espacio.
- `principal.bin` contiene un pangrama en español, lo que permitió fijar muchas
  letras por contexto.
- `largo.bin` y `noticia.bin` confirmaron el rango consecutivo `20..45 -> a..z`.
- Las fechas, horas y signos de puntuación de `noticia.bin` ayudaron a fijar
  dígitos y símbolos.
- Algunos bytes aparecían sistemáticamente después de letras, lo que permitió
  interpretarlos como modificadores y no como caracteres autónomos.

Flujo interno del programa
--------------------------
El programa se divide en tres capas:

1. Tablas de conversión:
   definen la correspondencia entre bytes y caracteres.

2. Conversión:
   - `decode_plncg26(data: bytes) -> str`
   - `encode_plncg26(text: str) -> bytes`

3. Interfaz de línea de comandos:
   `typer` interpreta el subcomando, lee el fichero de entrada y vuelca el
   resultado a `stdout` o `stdout.buffer`.

Resumen del algoritmo
---------------------
Decodificación:
- se lee un byte
- si es un símbolo simple, se emite directamente
- si es una letra base, se acumula temporalmente
- mientras haya modificadores a continuación, se aplican en orden
- el carácter resultante se añade a la salida

Codificación:
- se lee un carácter Unicode
- si es un símbolo simple, se emite su byte directo
- si es una letra con o sin diacríticos, se descompone en:
  letra base + marcas combinantes
- la base se transforma a su byte PLNCG26
- las marcas se traducen a bytes modificadores y se añaden detrás

Notas técnicas
--------------
- Los textos en PLNCG26 se tratan siempre como `bytes`, nunca como `str`.
- `decode` escribe texto Unicode en `stdout`.
- `encode` escribe binario en `stdout.buffer`.
- El script está pensado para ejecutarse con `uv run`, como pide el enunciado.
  Por ello no necesita permisos de ejecución extra (`chmod +x`) para cumplir
  la práctica.
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import typer

app = typer.Typer(
    add_completion=False,
    help=(
        "Conversor entre PLNCG26 y UTF-8 (Práctica P3 – Detectives de criptoglifos)\n\n"
        "Permite convertir ficheros entre el formato binario PLNCG26 y texto UTF-8.\n\n"
        "Comandos disponibles:\n"
        "  decode   Convierte PLNCG26 -> texto UTF-8\n"
        "  encode   Convierte texto UTF-8 -> PLNCG26\n"
        "  detect   Estima si un fichero parece estar en PLNCG26\n\n"
        "Ejemplos:\n"
        "  uv run fdi-pln-2608-p3.py decode principal.bin\n"
        "  uv run fdi-pln-2608-p3.py encode texto.txt > salida.bin\n"
        "  uv run fdi-pln-2608-p3.py detect principal.bin"
    ),
)

# ============================================================================
# TABLAS DE CONVERSIÓN
# ============================================================================

# Letras base: 20..45 -> a..z
BASE_DECODE: dict[int, str] = {
    20: "a",
    21: "b",
    22: "c",
    23: "d",
    24: "e",
    25: "f",
    26: "g",
    27: "h",
    28: "i",
    29: "j",
    30: "k",
    31: "l",
    32: "m",
    33: "n",
    34: "o",
    35: "p",
    36: "q",
    37: "r",
    38: "s",
    39: "t",
    40: "u",
    41: "v",
    42: "w",
    43: "x",
    44: "y",
    45: "z",
}

# Símbolos representados por un único byte.
SINGLE_DECODE: dict[int, str] = {
    10: "\n",
    11: " ",
    60: "0",
    61: "1",
    62: "2",
    63: "3",
    64: "4",
    65: "5",
    66: "6",
    67: "7",
    68: "8",
    69: "9",
    70: ".",
    71: ",",
    72: ";",
    73: ":",
    78: "-",
    79: "'",
    80: '"',
    81: "(",
    82: ")",
    100: "#",
    101: "*",
}

# Modificadores aplicados sobre la letra base inmediatamente anterior.
MOD_ACUTE = 50
MOD_DIAERESIS = 51
MOD_TILDE = 52
MOD_CAPITAL = 53

MODIFIERS = {MOD_ACUTE, MOD_DIAERESIS, MOD_TILDE, MOD_CAPITAL}

# Tablas inversas para codificar UTF-8 -> PLNCG26.
BASE_ENCODE: dict[str, int] = {char: code for code, char in BASE_DECODE.items()}
SINGLE_ENCODE: dict[str, int] = {char: code for code, char in SINGLE_DECODE.items()}

VALID_BYTES = set(BASE_DECODE) | set(SINGLE_DECODE) | MODIFIERS


def _error(message: str) -> None:
    """Escribe un mensaje de error por stderr."""
    print(f"Error: {message}", file=sys.stderr)


def _apply_modifier(char: str, modifier: int) -> str:
    """Aplica un modificador PLNCG26 sobre un carácter base."""
    if modifier == MOD_CAPITAL:
        return char.upper()
    if modifier == MOD_ACUTE:
        return unicodedata.normalize("NFC", char + "\u0301")
    if modifier == MOD_DIAERESIS:
        return unicodedata.normalize("NFC", char + "\u0308")
    if modifier == MOD_TILDE:
        return unicodedata.normalize("NFC", char + "\u0303")
    raise ValueError(f"Modificador desconocido: {modifier}")


def decode_plncg26(data: bytes) -> str:
    """Decodifica una secuencia de bytes PLNCG26 a texto Unicode."""
    output: list[str] = []
    index = 0

    while index < len(data):
        byte = data[index]

        # Símbolos simples: se emiten directamente.
        if byte in SINGLE_DECODE:
            output.append(SINGLE_DECODE[byte])
            index += 1
            continue

        # Letras base: pueden ir seguidas de varios modificadores.
        if byte in BASE_DECODE:
            char = BASE_DECODE[byte]
            index += 1

            while index < len(data) and data[index] in MODIFIERS:
                char = _apply_modifier(char, data[index])
                index += 1

            output.append(char)
            continue

        raise ValueError(f"Byte PLNCG26 desconocido en posición {index}: {byte}")

    return "".join(output)


def _decompose_char(char: str) -> tuple[str, list[int]]:
    """Descompone un carácter Unicode en base PLNCG26 y modificadores."""
    modifiers: list[int] = []

    # Las mayúsculas se codifican como:
    # letra minúscula base + modificador de mayúscula.
    if char.isalpha() and char.upper() == char and char.lower() != char:
        char = char.lower()
        modifiers.append(MOD_CAPITAL)

    # NFD separa por ejemplo "á" en "a" + acento combinante.
    nfd = unicodedata.normalize("NFD", char)
    base = nfd[0]

    for mark in nfd[1:]:
        if mark == "\u0301":
            modifiers.append(MOD_ACUTE)
        elif mark == "\u0308":
            modifiers.append(MOD_DIAERESIS)
        elif mark == "\u0303":
            modifiers.append(MOD_TILDE)
        else:
            codepoint = f"U+{ord(mark):04X}"
            raise ValueError(
                f"Diacrítico no soportado para PLNCG26: {codepoint} en {char!r}"
            )

    return base, modifiers


def encode_plncg26(text: str) -> bytes:
    """Codifica texto Unicode como bytes PLNCG26."""
    output = bytearray()

    for position, char in enumerate(text):
        if char in SINGLE_ENCODE:
            output.append(SINGLE_ENCODE[char])
            continue

        base, modifiers = _decompose_char(char)
        if base in BASE_ENCODE:
            output.append(BASE_ENCODE[base])
            output.extend(modifiers)
            continue

        raise ValueError(
            f"Carácter UTF-8 no soportado en posición {position}: {char!r}"
        )

    return bytes(output)


def detect_probability(data: bytes) -> float:
    """Estima heurísticamente si una secuencia parece pertenecer a PLNCG26."""
    if not data:
        return 0.0
    known = sum(1 for byte in data if byte in VALID_BYTES)
    return known / len(data)


@app.command()
def decode(
    file: Path = typer.Argument(
        ..., exists=True, readable=True, help="Fichero binario"
    ),
) -> None:
    """Convierte un fichero PLNCG26 a texto UTF-8."""
    try:
        data = file.read_bytes()
        text = decode_plncg26(data)
    except OSError as exc:
        _error(f"no se pudo leer {file}: {exc}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        _error(str(exc))
        raise typer.Exit(code=1) from exc

    sys.stdout.write(text)


@app.command()
def encode(
    file: Path = typer.Argument(
        ..., exists=True, readable=True, help="Fichero de texto UTF-8"
    ),
) -> None:
    """Convierte un fichero UTF-8 a PLNCG26."""
    try:
        text = file.read_text(encoding="utf-8")
        data = encode_plncg26(text)
    except OSError as exc:
        _error(f"no se pudo leer {file}: {exc}")
        raise typer.Exit(code=1) from exc
    except UnicodeDecodeError as exc:
        _error(f"{file} no está en UTF-8 válido: {exc}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        _error(str(exc))
        raise typer.Exit(code=1) from exc

    sys.stdout.buffer.write(data)


@app.command()
def detect(
    file: Path = typer.Argument(
        ..., exists=True, readable=True, help="Fichero a analizar"
    ),
) -> None:
    """Calcula una probabilidad heurística de que un fichero sea PLNCG26."""
    try:
        data = file.read_bytes()
    except OSError as exc:
        _error(f"no se pudo leer {file}: {exc}")
        raise typer.Exit(code=1) from exc

    probability = detect_probability(data)
    sys.stdout.write(f"{probability:.4f}\n")


if __name__ == "__main__":
    app()
