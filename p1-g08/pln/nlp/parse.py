"""Funciones de parseo de respuestas del LLM y ofertas de trueque.

Este módulo convierte texto libre o texto generado por el LLM en estructuras
simples que la lógica de negociación pueda usar de forma segura.
"""

import json
import re

from pln.nlp.normalize import normalizar_recurso


def parsear_json_llm(texto: str | None) -> dict:
    """Intenta extraer un JSON válido desde la respuesta del LLM.

    Primero prueba a interpretar todo el texto como JSON. Si falla,
    busca el primer bloque delimitado por llaves y vuelve a intentarlo.

    Si no se puede obtener JSON válido, devuelve un diccionario vacío.
    """
    if not texto:
        return {}

    texto = texto.strip()

    try:
        return json.loads(texto)
    except Exception:
        pass

    try:
        ini = texto.find("{")
        fin = texto.rfind("}")

        if ini != -1 and fin != -1 and fin > ini:
            return json.loads(texto[ini : fin + 1])

    except Exception:
        return {}

    return {}


def parse_oferta_v1(cuerpo: str | None) -> tuple[list[str], list[str]]:
    """Parsea una oferta en formato interno [OFERTA_V1].

    Extrae dos listas:
    - recursos que el remitente quiere,
    - recursos que el remitente ofrece.

    Si el formato no existe o no es válido, devuelve dos listas vacías.
    """
    if not cuerpo or "[OFERTA_V1]" not in cuerpo:
        return [], []

    m_quiero = re.search(r"quiero\s*=\s*(\{.*?\})", cuerpo)
    m_ofrezco = re.search(r"ofrezco\s*=\s*(\{.*?\})", cuerpo)

    quiere, ofrece = [], []

    try:
        if m_quiero:
            d = json.loads(m_quiero.group(1))
            quiere = [normalizar_recurso(k) for k in d.keys()]

        if m_ofrezco:
            d = json.loads(m_ofrezco.group(1))
            ofrece = [normalizar_recurso(k) for k in d.keys()]

    except Exception:
        return [], []

    return quiere, ofrece


def extraer_oferta_1x1_regex(texto: str | None) -> tuple[str | None, str | None]:
    """Extrae una oferta 1x1 usando patrones regulares simples.

    Detecta mensajes como:
    - te doy 1 madera a cambio de 1 piedra
    - ofrezco 1 oro por 1 trigo
    - cambio 1 hierro por 1 agua

    Devuelve:
    - recurso ofrecido,
    - recurso solicitado.

    Si no encuentra una oferta clara, devuelve None, None.
    """
    if not texto:
        return None, None

    t = " ".join(texto.lower().split())

    patrones = [
        r"te\s+doy\s+1\s+(\w+)\s+a\s+cambio\s+de\s+1\s+(\w+)",
        r"ofrezco\s*1\s+(\w+)\s+por\s+1\s+(\w+)",
        r"cambio\s+1\s+(\w+)\s+por\s+1\s+(\w+)",
        r"1\s+(\w+)\s+por\s+1\s+(\w+)",
    ]

    for pat in patrones:
        m = re.search(pat, t)

        if m:
            return normalizar_recurso(m.group(1)), normalizar_recurso(m.group(2))

    return None, None
