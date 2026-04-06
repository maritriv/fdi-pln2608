"""Modulo de Generacion Aumentada por Recuperacion (RAG)."""

from __future__ import annotations

import ollama

from quijote_app.config import DEFAULT_LLM_MODEL
from quijote_app.models import PassageIndex, SearchResult
from quijote_app.search import search_passages, search_semantic_passages


def _format_reference(result: SearchResult) -> str:
    """Devuelve una referencia legible para el usuario."""
    passage = result.passage
    chapter = (passage.chapter or "").strip()

    if chapter:
        return f"{chapter} [{passage.passage_id}]"

    return f"Pasaje [{passage.passage_id}]"


def generate_answer(
    index: PassageIndex,
    query: str,
    mode: str = "semantic",
    model: str = DEFAULT_LLM_MODEL,
    limit: int = 3,
    chapter_filter: str | None = None,
) -> str:
    """Recupera contexto y genera una respuesta usando un LLM."""

    semantic_results = search_semantic_passages(
        index, query, limit=limit, chapter_filter=chapter_filter
    )
    classic_results = search_passages(
        index, query, limit=limit, chapter_filter=chapter_filter
    )

    merged = []
    seen_ids = set()

    for result in semantic_results + classic_results:
        if result.passage.passage_id not in seen_ids:
            merged.append(result)
            seen_ids.add(result.passage.passage_id)

    results = merged[:limit]

    if not results:
        return "No he encontrado informacion suficiente en el Quijote para responder a eso."

    context_text = "\n\n".join(
        [
            f"REFERENCIA: {_format_reference(r)}\nTEXTO: {r.passage.text_original}"
            for r in results
        ]
    )

    prompt = f"""
Eres un experto en "El ingenioso hidalgo don Quijote de la Mancha".

Responde usando UNICAMENTE la informacion contenida en los pasajes.
Si la respuesta no aparece claramente en el contexto, dilo.
No inventes datos.
Responde de forma breve, clara y directa, en un solo parrafo.
NO anadas referencias, citas, listas ni identificadores al final.
Solo devuelve la respuesta redactada.

PASAJES DE REFERENCIA:
{context_text}

PREGUNTA DEL USUARIO:
{query}

RESPUESTA:
"""

    try:
        response = ollama.generate(model=model, prompt=prompt)
        answer = response.get("response", "Error al generar respuesta.").strip()

        references = "\n".join(f"- {_format_reference(r)}" for r in results)
        return f"{answer}\n\nReferencias:\n{references}"

    except Exception as e:
        return (
            "Error de conexion con Ollama: Verifica que esta abierto y el modelo "
            f"descargado. Detalle: {e}"
        )
