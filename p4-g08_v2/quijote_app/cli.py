"""CLI principal basada en Typer."""

from __future__ import annotations

from collections import Counter
import re
from pathlib import Path
from statistics import mean
import sys
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from quijote_app.config import DEFAULT_LIMIT, MIN_WORDS_PER_PASSAGE, SNIPPET_MAX_CHARS
from quijote_app.corpus import CorpusError, resolve_source_path
from quijote_app.indexing import IndexError, load_or_build_index
from quijote_app.models import SearchResult
from quijote_app.search import search_passages, search_semantic_passages
from quijote_app.rag import generate_answer
from quijote_app.utils import render_excerpt

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Buscador clasico por lemas del Quijote (recuperaciÃ³n de pasajes reales).",
)

EXIT_COMMANDS = {"exit", "quit", "salir", "/exit", "/quit", "/salir"}
console = Console()
HELP_COMMANDS = {"/help", "help", "ayuda", "/ayuda"}
STATS_COMMANDS = {"/stats", "stats", "estado", "/estado"}
HIGHLIGHT_STYLE = "bold magenta"
VALID_MODES = {"classic", "semantic", "rag"}


@app.command("index")
def index_command(
    source: Optional[Path] = typer.Option(
        None,
        "--source",
        "-s",
        help="Ruta a ZIP o HTML del corpus.",
    ),
    cache_path: Optional[Path] = typer.Option(
        None,
        "--cache-path",
        help="Ruta de caché del í­dice (pickle).",
    ),
    min_words: int = typer.Option(
        MIN_WORDS_PER_PASSAGE,
        "--min-words",
        min=1,
        help="Mínimo de palabras por pasaje indexable.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Reconstruye índice aunque exista caché válida.",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="No leer ni guardar caché.",
    ),
) -> None:
    """Construye o actualiza el índice del corpus."""
    try:
        resolved_source = resolve_source_path(source)
        index, from_cache, effective_cache = load_or_build_index(
            source=resolved_source,
            cache_path=cache_path,
            use_cache=not no_cache,
            force_rebuild=force,
            min_words=min_words,
        )
    except (CorpusError, IndexError, ValueError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Fuente: {resolved_source}")
    if index.metadata.selected_entry:
        typer.echo(f"Entrada HTML (ZIP): {index.metadata.selected_entry}")
    if no_cache:
        typer.echo("Caché: desactivada")
    else:
        status = "reutilizada" if from_cache else "actualizada"
        typer.echo(f"Caché: {effective_cache} ({status})")

    typer.echo(f"Pasajes indexados: {index.metadata.passage_count}")
    typer.echo(f"Capítulos detectados: {index.metadata.chapter_count}")
    typer.echo(f"Partes detectadas: {index.metadata.part_count}")


@app.command("search")
def search_command(
    query: str = typer.Argument(..., help="Término o frase a buscar."),
    mode: str = typer.Option(
        "classic", "--mode", "-m", help="Modo: classic, semantic, rag"
    ),  # <-- ESTA LÍNEA NUEVA
    source: Optional[Path] = typer.Option(
        None,
        "--source",
        "-s",
        help="Ruta a ZIP o HTML del corpus.",
    ),
    limit: int = typer.Option(
        DEFAULT_LIMIT,
        "--limit",
        "-l",
        min=1,
        max=100,
        help="Máximo de resultados a mostrar.",
    ),
    chapter: Optional[str] = typer.Option(
        None,
        "--chapter",
        help="Filtro textual por capítulo (normalizado).",
    ),
    cache_path: Optional[Path] = typer.Option(
        None,
        "--cache-path",
        help="Ruta de cachÃ© del índice (pickle).",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="No usar caché de índice.",
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Reindexa antes de buscar.",
    ),
    max_chars: int = typer.Option(
        SNIPPET_MAX_CHARS,
        "--max-chars",
        min=120,
        max=1200,
        help="Longitud máxima de pasaje mostrado.",
    ),
) -> None:
    """Busca y devuelve pasajes relevantes del corpus usando lemas."""
    if not query.strip():
        typer.secho(
            "Error: la consulta no puede estar vacía.", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)

    try:
        resolved_source = resolve_source_path(source)
        index, from_cache, effective_cache = load_or_build_index(
            source=resolved_source,
            cache_path=cache_path,
            use_cache=not no_cache,
            force_rebuild=rebuild,
        )
        if mode not in VALID_MODES:
            raise ValueError(f"Modo inválido. Elige entre: {VALID_MODES}")

        if mode == "rag":
            console.print("\n[dim]Pensando respuesta con IA...[/dim]")
            answer = generate_answer(
                index, query, mode="semantic", limit=limit, chapter_filter=chapter
            )
            console.print()
            console.print(
                Panel(
                    answer,
                    title="[bold green]Respuesta RAG[/bold green]",
                    border_style="green",
                )
            )
            return  # El RAG no imprime lista de pasajes

        if mode == "semantic":
            results = search_semantic_passages(index, query, limit, chapter)
        else:
            results = search_passages(index, query, limit, chapter)

    except (CorpusError, IndexError, ValueError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    summary = Table(box=box.SIMPLE_HEAVY, show_header=False)
    summary.add_column("Campo", style="bold cyan")
    summary.add_column("Valor")
    summary.add_row("Consulta", query.strip())
    summary.add_row("Modo", mode.upper())
    summary.add_row("Resultados", str(len(results)))
    summary.add_row("Fuente", str(resolved_source))
    if chapter:
        summary.add_row("Filtro capitulo", chapter)
    if no_cache:
        summary.add_row("Cache", "desactivada")
    else:
        cache_status = "reutilizada" if from_cache else "actualizada"
        summary.add_row("Cache", f"{cache_status} -> {effective_cache}")
    console.print(Panel(summary, title="[bold]Resumen[/bold]", border_style="cyan"))
    console.print()

    _print_results(
        query=query,
        results=results,
        max_chars=max_chars,
        is_semantic=(mode == "semantic"),
    )


@app.command("interactive")
def interactive_command(
    source: Optional[Path] = typer.Option(
        None,
        "--source",
        "-s",
        help="Ruta a ZIP o HTML del corpus.",
    ),
    limit: int = typer.Option(
        DEFAULT_LIMIT,
        "--limit",
        "-l",
        min=1,
        max=100,
        help="Máximo de resultados por consulta.",
    ),
    chapter: Optional[str] = typer.Option(
        None,
        "--chapter",
        help="Filtro textual inicial por capítulo (normalizado).",
    ),
    cache_path: Optional[Path] = typer.Option(
        None,
        "--cache-path",
        help="Ruta de caché del índice (pickle).",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="No usar caché de índice.",
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Reindexa al iniciar la sesión.",
    ),
    max_chars: int = typer.Option(
        SNIPPET_MAX_CHARS,
        "--max-chars",
        min=120,
        max=1200,
        help="Longitud máxima de pasaje mostrado.",
    ),
) -> None:
    """Lanza una sesión interactiva de consultas hasta escribir exit."""
    _run_interactive_session(
        source=source,
        limit=limit,
        chapter=chapter,
        cache_path=cache_path,
        no_cache=no_cache,
        rebuild=rebuild,
        max_chars=max_chars,
    )


def _run_interactive_session(
    source: Optional[Path],
    limit: int,
    chapter: Optional[str],
    cache_path: Optional[Path],
    no_cache: bool,
    rebuild: bool,
    max_chars: int,
) -> None:
    try:
        resolved_source = resolve_source_path(source)
        index, from_cache, effective_cache = load_or_build_index(
            source=resolved_source,
            cache_path=cache_path,
            use_cache=not no_cache,
            force_rebuild=rebuild,
        )
    except (CorpusError, IndexError, ValueError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    current_limit = limit
    current_chapter = chapter
    current_mode = "classic"  # <-- NUEVO
    cache_status = (
        "desactivada" if no_cache else ("reutilizada" if from_cache else "actualizada")
    )

    welcome = Text()
    welcome.append("Hola, soy Quijote App.\n", style="bold cyan")
    welcome.append("Buscador multimodelo del Quijote para consultas en terminal.\n")
    welcome.append(
        "Modos disponibles: Clásico (lemas), Semántico (embeddings) y RAG.\n\n"
    )
    welcome.append(f"Fuente cargada: {resolved_source}\n", style="dim")
    if no_cache:
        welcome.append("Cache: desactivada\n", style="dim")
    else:
        welcome.append(f"Cache: {cache_status} -> {effective_cache}\n", style="dim")
    welcome.append("\nSi es tu primera vez, ejecuta ", style="bold")
    welcome.append("/help", style="bold cyan")
    welcome.append(" para ver comandos con ejemplos.\n")
    console.print(
        Panel.fit(welcome, title="[bold]Sesion interactiva[/bold]", border_style="cyan")
    )
    _print_quickstart_guide()
    console.print()

    while True:
        try:
            # Aquí inyectamos el modo en el prompt de la terminal
            raw = console.input(f"[bold cyan]quijote ({current_mode})> [/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        query = raw.strip()
        if not query:
            continue

        lowered = query.lower()
        if lowered in EXIT_COMMANDS:
            break

        if lowered in HELP_COMMANDS:
            _print_interactive_help()
            continue

        if lowered in STATS_COMMANDS:
            _print_interactive_stats(
                source=resolved_source,
                passage_count=len(index.passages),
                chapter_count=index.metadata.chapter_count,
                current_limit=current_limit,
                current_chapter=current_chapter,
            )
            continue

        if lowered.startswith("/limit ") or lowered.startswith("limit "):
            value = query.split(maxsplit=1)[1].strip() if " " in query else ""
            try:
                parsed_limit = int(value)
            except ValueError:
                console.print(
                    "[yellow]Valor de /limit invalido. Debe ser un entero.[/yellow]"
                )
                continue

            if parsed_limit < 1 or parsed_limit > 100:
                console.print("[yellow]El limite debe estar entre 1 y 100.[/yellow]")
                continue

            current_limit = parsed_limit
            console.print(f"[green]Limite actualizado:[/green] {current_limit}")
            continue

        if lowered.startswith("/chapter ") or lowered.startswith("chapter "):
            chapter_value = query.split(maxsplit=1)[1].strip() if " " in query else ""
            if not chapter_value:
                console.print(
                    "[yellow]Uso: /chapter capitulo xxv  (o /chapter off para quitar filtro).[/yellow]"
                )
                continue
            if chapter_value.lower() in {"off", "none", "clear", "ninguno"}:
                current_chapter = None
                console.print("[green]Filtro de capitulo desactivado.[/green]")
            else:
                current_chapter = chapter_value
                console.print(
                    f"[green]Filtro de capitulo activo:[/green] {current_chapter}"
                )
            continue

        # --- AQUÍ AÑADIMOS EL CAMBIO DE MODO ---
        if lowered.startswith("/mode ") or lowered.startswith("mode "):
            new_mode = (
                query.split(maxsplit=1)[1].strip().lower() if " " in query else ""
            )
            if new_mode in VALID_MODES:
                current_mode = new_mode
                console.print(
                    f"[green]Modo cambiado a:[/green] [bold]{current_mode.upper()}[/bold]"
                )
            else:
                console.print(
                    f"[yellow]Modo inválido. Usa: {', '.join(VALID_MODES)}[/yellow]"
                )
            continue

        if query.startswith("/"):
            console.print(
                "[yellow]Comando no reconocido. Usa /help para ver comandos validos.[/yellow]"
            )
            continue

        # --- AQUÍ ESTÁ LA NUEVA EJECUCIÓN DE LA BÚSQUEDA ---
        try:
            if current_mode == "rag":
                console.print("[dim]Pensando respuesta con IA...[/dim]")
                answer = generate_answer(
                    index,
                    query,
                    mode="semantic",
                    limit=current_limit,
                    chapter_filter=current_chapter,
                )
                console.print(
                    Panel(
                        answer,
                        title="[bold green]Respuesta RAG[/bold green]",
                        border_style="green",
                    )
                )
                continue  # Si es RAG, ya hemos respondido, volvemos a pedir input al usuario

            elif current_mode == "semantic":
                results = search_semantic_passages(
                    index, query, current_limit, current_chapter
                )
            else:
                results = search_passages(index, query, current_limit, current_chapter)

        except ValueError as exc:
            console.print(f"[yellow]Consulta invalida:[/yellow] {exc}")
            continue

        console.print()
        console.print(f"[bold]Consulta:[/bold] {query}")
        console.print(f"[bold]Resultados:[/bold] {len(results)}")
        if current_chapter:
            console.print(f"[bold]Filtro capitulo activo:[/bold] {current_chapter}")
        console.print()

        # --- AQUÍ LE PASAMOS EL PARÁMETRO is_semantic ---
        _print_results(
            query=query,
            results=results,
            max_chars=max_chars,
            is_semantic=(current_mode == "semantic"),
        )

    console.print("[bold]Sesion finalizada.[/bold]")


@app.command("stats")
def stats_command(
    source: Optional[Path] = typer.Option(
        None,
        "--source",
        "-s",
        help="Ruta a ZIP o HTML del corpus.",
    ),
    cache_path: Optional[Path] = typer.Option(
        None,
        "--cache-path",
        help="Ruta de caché del índice (pickle).",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="No usar caché de índice.",
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Forzar reconstrucción del índice.",
    ),
) -> None:
    """Muestra estadísticas del Ãíndice y del corpus."""
    try:
        resolved_source = resolve_source_path(source)
        index, from_cache, effective_cache = load_or_build_index(
            source=resolved_source,
            cache_path=cache_path,
            use_cache=not no_cache,
            force_rebuild=rebuild,
        )
    except (CorpusError, IndexError, ValueError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    passages = index.passages
    avg_words = (
        mean(len(p.text_normalized.split()) for p in passages) if passages else 0.0
    )
    unique_chapters = len({p.chapter for p in passages if p.chapter})

    typer.echo("Estadísticas del índice")
    typer.echo("----------------------")
    typer.echo(f"Fuente: {index.metadata.source_path}")
    typer.echo(f"Tipo de fuente: {index.metadata.source_kind}")
    if index.metadata.selected_entry:
        typer.echo(f"Entrada HTML (ZIP): {index.metadata.selected_entry}")
    typer.echo(f"Pasajes indexados: {len(passages)}")
    typer.echo(f"Capítulos detectados (metadato): {index.metadata.chapter_count}")
    typer.echo(f"Capítulos con pasajes: {unique_chapters}")
    typer.echo(f"Partes detectadas: {index.metadata.part_count}")
    typer.echo(f"Media de palabras por pasaje: {avg_words:.2f}")
    typer.echo(f"Fecha construcción (UTC): {index.built_at_iso}")
    if no_cache:
        typer.echo("Caché: desactivada")
    else:
        cache_status = "reutilizada" if from_cache else "actualizada"
        typer.echo(f"Caché: {effective_cache} ({cache_status})")


@app.command("chapters")
def chapters_command(
    source: Optional[Path] = typer.Option(
        None,
        "--source",
        "-s",
        help="Ruta a ZIP o HTML del corpus.",
    ),
    cache_path: Optional[Path] = typer.Option(
        None,
        "--cache-path",
        help="Ruta de caché del índice (pickle).",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        min=1,
        help="NÃºmero mÃ¡ximo de capítulos a listar.",
    ),
) -> None:
    """Lista capítulos detectados y nÃºmero de pasajes por capítulo."""
    try:
        resolved_source = resolve_source_path(source)
        index, _, _ = load_or_build_index(
            source=resolved_source,
            cache_path=cache_path,
            use_cache=True,
            force_rebuild=False,
        )
    except (CorpusError, IndexError, ValueError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    counter = Counter((p.chapter or "Sin capítulo detectado") for p in index.passages)
    typer.echo(f"Capí­tulos listados: {min(limit, len(counter))}/{len(counter)}")
    typer.echo("")
    for idx, (chapter_name, count) in enumerate(counter.most_common(limit), start=1):
        typer.echo(f"{idx:>3}. {_shorten(chapter_name, 140)} ({count} pasajes)")


def _print_results(
    query: str, results: list[SearchResult], max_chars: int, is_semantic: bool = False
) -> None:
    if not results:
        console.print("[yellow]No se encontraron pasajes para esa consulta.[/yellow]")
        return

    for idx, result in enumerate(results, start=1):
        chapter_label = _shorten(
            result.passage.chapter or "Sin capítulo detectado", 120
        )
        excerpt = render_excerpt(
            result.passage.text_original, query=query, max_chars=max_chars
        )

        if is_semantic:
            meta = f"similitud del coseno={result.score:.4f}"
        else:
            meta = (
                f"score={result.score:.2f} | exactas={result.exact_matches} "
                f"| hits={result.total_term_hits}"
            )

        content = Text()
        content.append(meta + "\n", style="dim")
        content.append_text(_excerpt_to_rich_text(excerpt))
        console.print(
            Panel(
                content,
                title=f"[bold cyan][{idx}][/bold cyan] {chapter_label}",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )


def _excerpt_to_rich_text(excerpt: str) -> Text:
    """Convierte marcas [coincidencia] en texto coloreado para terminal."""
    rich_excerpt = Text()
    cursor = 0
    for match in re.finditer(r"\[([^\]]+)\]", excerpt):
        start, end = match.span()
        if start > cursor:
            rich_excerpt.append(excerpt[cursor:start])
        rich_excerpt.append(match.group(1), style=HIGHLIGHT_STYLE)
        cursor = end

    if cursor < len(excerpt):
        rich_excerpt.append(excerpt[cursor:])

    return rich_excerpt


def _print_interactive_help() -> None:
    help_table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan")
    help_table.add_column("Comando", style="bold")
    help_table.add_column("Que hace")
    help_table.add_column("Ejemplo")
    help_table.add_row(
        "consulta libre",
        "Busca pasajes por lema (palabra o frase)",
        '"sancho panza" o "domingo"',
    )
    help_table.add_row(
        "/mode TIPO", "Cambia el modo a classic, semantic o rag", "/mode rag"
    )
    help_table.add_row("/help", "Muestra esta ayuda", "/help")
    help_table.add_row("/limit N", "Define cuantos resultados ver (1..100)", "/limit 5")
    help_table.add_row(
        "/chapter TEXTO", "Filtra resultados por capitulo", "/chapter capitulo xxv"
    )
    help_table.add_row("/chapter off", "Quita el filtro de capitulo", "/chapter off")
    help_table.add_row("/stats", "Muestra estado de la sesion", "/stats")
    help_table.add_row("exit | quit | salir", "Cierra la sesion", "exit")
    console.print(
        Panel(help_table, title="[bold]Ayuda de sesion[/bold]", border_style="cyan")
    )
    console.print()


def _print_quickstart_guide() -> None:
    quickstart = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan")
    quickstart.add_column("Cosa que puedes hacer")
    quickstart.add_column("Que escribir")
    quickstart.add_column("Que obtienes")
    quickstart.add_row(
        "Buscar un personaje",
        '"sancho panza"',
        "Pasajes donde aparece Sancho Panza.",
    )
    quickstart.add_row(
        "Buscar a Dulcinea",
        '"dulcinea del toboso"',
        "Dulcinea es el amor idealizado de Don Quijote.",
    )
    quickstart.add_row(
        "Cambiar la magia de búsqueda",
        "/mode rag",
        "Activa la IA para generar respuestas (RAG).",
    )
    quickstart.add_row(
        "Buscar una fecha o referencia temporal",
        '"domingo"',
        "Pasajes con referencias temporales.",
    )
    quickstart.add_row(
        "Ver mas o menos resultados",
        "/limit 3",
        "A partir de ahora muestra 3 resultados por consulta.",
    )
    quickstart.add_row(
        "Filtrar por capitulo",
        "/chapter capitulo xxv",
        "Busca solo dentro de ese capitulo.",
    )
    quickstart.add_row(
        "Quitar filtro de capitulo",
        "/chapter off",
        "Vuelve a buscar en todo el libro.",
    )
    quickstart.add_row(
        "Ver estado actual",
        "/stats",
        "Muestra limite activo, filtro y datos de la sesion.",
    )
    quickstart.add_row(
        "Ver ayuda completa",
        "/help",
        "Muestra todos los comandos con ejemplos.",
    )
    quickstart.add_row(
        "Salir",
        "exit",
        "Finaliza la sesion interactiva.",
    )
    console.print(
        Panel(
            quickstart, title="[bold]Cosas que puedes hacer[/bold]", border_style="cyan"
        )
    )


def _print_interactive_stats(
    source: Path,
    passage_count: int,
    chapter_count: int,
    current_limit: int,
    current_chapter: Optional[str],
) -> None:
    stats_table = Table(box=box.SIMPLE_HEAVY, show_header=False)
    stats_table.add_column("Campo", style="bold cyan")
    stats_table.add_column("Valor")
    stats_table.add_row("Fuente", str(source))
    stats_table.add_row("Pasajes indexados", str(passage_count))
    stats_table.add_row("Capitulos detectados", str(chapter_count))
    stats_table.add_row("Limite actual", str(current_limit))
    stats_table.add_row("Filtro capitulo", current_chapter or "ninguno")
    console.print(
        Panel(stats_table, title="[bold]Estado de sesion[/bold]", border_style="cyan")
    )
    console.print()


def _shorten(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def run() -> None:
    """Punto de entrada para script de consola."""
    if len(sys.argv) == 1:
        _run_interactive_session(
            source=None,
            limit=DEFAULT_LIMIT,
            chapter=None,
            cache_path=None,
            no_cache=False,
            rebuild=False,
            max_chars=SNIPPET_MAX_CHARS,
        )
        return
    app()
