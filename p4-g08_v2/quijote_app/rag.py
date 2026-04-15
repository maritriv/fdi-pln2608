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
        short = chapter if len(chapter) <= 90 else chapter[:87].rstrip() + "..."
        return f"{short} [{passage.passage_id}]"

    return f"Pasaje [{passage.passage_id}]"


def _score_for_rag(query: str, result: SearchResult) -> float:
    """Reordena resultados para priorizar pasajes mas utiles en RAG."""
    text = result.passage.text_original.lower()
    chapter = (result.passage.chapter or "").lower()
    query_terms = [term for term in query.lower().split() if len(term) > 2]

    score = float(result.score)

    exact_hits = sum(1 for term in query_terms if term in text)
    score += exact_hits * 0.20

    definitional_markers = (
        "es ",
        "era ",
        "fue ",
        "dama",
        "escudero",
        "amada",
        "senora",
        "senor",
        "hidalgo",
        "caballero",
        "toboso",
        "dulcinea",
        "sancho panza",
    )
    if any(marker in text for marker in definitional_markers):
        score += 0.20

    if "prologo" in chapter:
        score -= 0.10

    return score


def generate_answer(
    index: PassageIndex,
    query: str,
    mode: str = "semantic",
    model: str = DEFAULT_LLM_MODEL,
    limit: int = 4,
    chapter_filter: str | None = None,
) -> str:
    """Recupera contexto y genera una respuesta usando un LLM."""
    retrieval_limit = max(limit * 3, 8)

    semantic_results = search_semantic_passages(
        index,
        query,
        limit=retrieval_limit,
        chapter_filter=chapter_filter,
    )
    classic_results = search_passages(
        index,
        query,
        limit=retrieval_limit,
        chapter_filter=chapter_filter,
    )

    merged: list[SearchResult] = []
    seen_ids: set[str] = set()

    for result in classic_results + semantic_results:
        if result.passage.passage_id not in seen_ids:
            merged.append(result)
            seen_ids.add(result.passage.passage_id)

    ranked = sorted(
        merged,
        key=lambda result: _score_for_rag(query, result),
        reverse=True,
    )
    results = ranked[:limit]

    if not results:
        return "No he encontrado informacion suficiente en el Quijote para responder a eso."

    context_text = "\n\n".join(
        [
            f"REFERENCIA: {_format_reference(result)}\n"
            f"TEXTO: {result.passage.text_original}"
            for result in results
        ]
    )

    prompt = f"""
Eres un asistente experto en "El ingenioso hidalgo don Quijote de la Mancha".

Tu tarea es responder usando UNICAMENTE la informacion contenida en los pasajes proporcionados.

Reglas:
- No inventes datos.
- No uses conocimiento externo.
- Si la respuesta no aparece con claridad en los pasajes, di exactamente:
  "No he encontrado informacion suficiente en los pasajes recuperados."
- Responde de forma breve, clara y directa.
- Si la pregunta es del tipo "quien es..." o "que relacion tiene...",
  empieza con una definicion corta y luego añade una o dos frases de apoyo.
- No incluyas referencias, listas, identificadores ni citas al final.
- Devuelve solo la respuesta redactada en texto corrido.

PASAJES DE REFERENCIA:
{context_text}

PREGUNTA DEL USUARIO:
{query}

RESPUESTA:
"""

    try:
        response = ollama.generate(model=model, prompt=prompt)
        answer = response.get("response", "Error al generar respuesta.").strip()

        references = "\n".join(f"- {_format_reference(result)}" for result in results)
        return f"{answer}\n\nReferencias:\n{references}"

    except Exception as e:
        return (
            "Error de conexion con Ollama: Verifica que esta abierto y el modelo "
            f"descargado. Detalle: {e}"
        )
