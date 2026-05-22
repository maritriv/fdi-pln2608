"""Utilidades para interacción con el modelo LLM mediante Ollama.

Este módulo encapsula las llamadas al modelo de lenguaje local y
la interpretación semántica de cartas recibidas durante la simulación.

La salida del LLM se restringe a JSON estructurado para facilitar
el procesamiento automático de trueques y recursos.
"""

import httpx

from pln.config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT
from pln.logger import log
from pln.nlp.parse import parsear_json_llm
from pln.nlp.normalize import normalizar_recurso


def preguntar_llm(prompt: str) -> str:
    """Envía un prompt al modelo LLM configurado en Ollama.

    Devuelve únicamente el texto generado por el modelo.
    Si ocurre un error de red o del modelo, devuelve una cadena vacía.
    """
    try:
        r = httpx.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=OLLAMA_TIMEOUT,
        )
        r.raise_for_status()

        return r.json().get("response", "")

    except Exception as e:
        log(f"LLM error: {e}")
        return ""


def interpretar_carta_a_listas(carta: dict) -> dict:
    """Extrae recursos ofrecidos y solicitados desde una carta.

    El LLM analiza el contenido textual de la carta y devuelve
    una estructura JSON con dos listas:
    - quiere: recursos que el remitente solicita
    - ofrece: recursos que el remitente está dispuesto a intercambiar

    Los recursos extraídos se normalizan para facilitar comparaciones
    posteriores dentro de la lógica de negociación.
    """
    prompt = f"""
[ROLE: SYSTEM]
Eres un extractor de trueques. NO negocias, NO redactas respuestas.
Solo extraes recursos del texto, aunque esté informal o con listas.

Devuelve SOLO JSON válido (sin texto extra) con:
{{ "quiere": ["..."], "ofrece": ["..."] }}

Reglas:
- No inventes recursos.
- Si hay cantidades, ignóralas (solo nombres).
- "acepto otros recursos" no añade nada si no se listan.
- "tengo X" cuenta como OFRECE solo si sugiere intercambio (por/a cambio/intercambio).

CARTA:
Asunto: {carta.get("asunto")}
Contenido:
{carta.get("cuerpo")}
"""

    raw = preguntar_llm(prompt)

    data = parsear_json_llm(raw)

    quiere = [
        normalizar_recurso(x) for x in data.get("quiere", []) if normalizar_recurso(x)
    ]

    ofrece = [
        normalizar_recurso(x) for x in data.get("ofrece", []) if normalizar_recurso(x)
    ]

    return {
        "quiere": quiere,
        "ofrece": ofrece,
    }
