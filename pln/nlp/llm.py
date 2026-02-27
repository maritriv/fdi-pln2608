import httpx
from pln.config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT
from pln.logger import log
from pln.nlp.parse import parsear_json_llm
from pln.nlp.normalize import normalizar_recurso


def preguntar_llm(prompt: str) -> str:
    try:
        r = httpx.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        log(f"LLM error: {e}")
        return ""


def interpretar_carta_a_listas(carta):
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
    return {"quiere": quiere, "ofrece": ofrece}
