"""Módulo de Generación Aumentada por Recuperación (RAG)."""

from __future__ import annotations
import ollama

from quijote_app.config import DEFAULT_LLM_MODEL
from quijote_app.models import PassageIndex
from quijote_app.search import search_semantic_passages, search_passages

def generate_answer(
    index: PassageIndex,
    query: str,
    mode: str = "semantic",
    model: str = DEFAULT_LLM_MODEL,
    limit: int = 3,
    chapter_filter: str | None = None
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
        return "No he encontrado información suficiente en el Quijote para responder a eso."

    context_text = "\n\n".join(
        [
            f"[{r.passage.passage_id}] {r.passage.chapter or 'Capítulo desconocido'}: {r.passage.text_original}"
            for r in results
        ]
    )

    prompt = f"""
Eres un experto en "El ingenioso hidalgo don Quijote de la Mancha".

Responde usando ÚNICAMENTE la información contenida en los pasajes.
Si la respuesta no aparece claramente en el contexto, dilo.
No inventes datos.
Al final, cita entre corchetes los identificadores de los pasajes que has usado.

PASAJES DE REFERENCIA:
{context_text}

PREGUNTA DEL USUARIO:
{query}

RESPUESTA:
"""

    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response.get("response", "Error al generar respuesta.")
    except Exception as e:
        return f"Error de conexión con Ollama: Verifica que está abierto y el modelo descargado. Detalle: {e}"