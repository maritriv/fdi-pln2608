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
    
    # 1. Recuperamos los pasajes más relevantes como contexto
    if mode == "semantic":
        results = search_semantic_passages(index, query, limit=limit, chapter_filter=chapter_filter)
    else:
        results = search_passages(index, query, limit=limit, chapter_filter=chapter_filter)
    
    if not results:
        return "No he encontrado información suficiente en el Quijote para responder a eso."

    # 2. Construimos el contexto con los pasajes encontrados
    context_text = "\n\n".join([
        f"Capítulo {r.passage.chapter or 'Desconocido'}: {r.passage.text_original}"
        for r in results
    ])

    # 3. Diseñamos el Prompt
    prompt = f"""
    Eres un experto en "El ingenioso hidalgo don Quijote de la Mancha".
    Usa ÚNICAMENTE los siguientes pasajes del libro para responder a la pregunta. 
    Si la respuesta no está en el texto, di claramente que no lo sabes basándote en el contexto.
    Cita siempre el capítulo o la parte de donde sacas la información.

    PASAJES DE REFERENCIA:
    {context_text}

    PREGUNTA DEL USUARIO:
    {query}
    
    RESPUESTA:
    """

    # 4. Llamamos a Ollama
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response.get("response", "Error al generar respuesta.")
    except Exception as e:
        return f"Error de conexión con Ollama: Verifica que está abierto y el modelo descargado. Detalle: {e}"