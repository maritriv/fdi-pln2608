"""Helpers visuales compartidos por el CLI Typer/Rich."""

from pathlib import Path

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


console = Console()


def render_header() -> None:
    title = Text("P5 · Transformer + NER", style="bold cyan")
    subtitle = Text("Grupo 2608", style="bold white")
    body = Text.assemble(
        title,
        "\n",
        subtitle,
        "\n\n",
        ("Mini modelo Transformer causal + NER BIO\n", "white"),
        ("implementado desde cero para PLN.", "white"),
    )
    console.print(Panel.fit(body, box=box.DOUBLE, border_style="cyan", padding=(1, 4)))


def render_section(title: str, description: str | None = None) -> None:
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))
    if description:
        console.print(f"[dim]{description}[/dim]\n")


def render_success(message: str) -> None:
    console.print(f"[green]OK[/green] {message}")


def render_error(message: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {message}")


def pause() -> None:
    Prompt.ask(
        "\n[dim]Pulsa Enter para volver al menú[/dim]",
        default="",
        show_default=False,
    )


def ask_with_default(label: str, default: str) -> str:
    value = Prompt.ask(
        f"{label} [cyan]\\[{escape(default)}][/cyan]",
        default="",
        show_default=False,
    )
    return value.strip() or default


def ask_int_with_default(label: str, default: int) -> int:
    while True:
        value = ask_with_default(label, str(default))
        try:
            return int(value)
        except ValueError:
            render_error("Introduce un número entero válido.")


def ask_float_with_default(label: str, default: float) -> float:
    while True:
        value = ask_with_default(label, str(default))
        try:
            return float(value)
        except ValueError:
            render_error("Introduce un número decimal válido.")


def path_exists(path: str) -> bool:
    if Path(path).exists():
        return True
    render_error(f"No se ha encontrado el fichero indicado: {path}")
    return False


def render_menu() -> None:
    render_header()
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column(style="white")
    table.add_row("[1]", "Generar texto")
    table.add_row("[2]", "Detectar entidades NER en un fichero")
    table.add_row("[3]", "Evaluar el modelo NER")
    table.add_row("[4]", "Analizar tokenización BPE")
    table.add_row("[5]", "Ver comandos disponibles")
    table.add_row("[0]", "Salir")
    console.print("\n[bold]¿Qué quieres hacer?[/bold]\n")
    console.print(table)
    console.print("\n[dim]Escribe una opción y pulsa Enter.[/dim]")


def render_entities_table(entities: list[tuple[str, str]]) -> None:
    table = Table(title="Entidades detectadas", box=box.SIMPLE_HEAVY)
    table.add_column("Entidad", style="bold white")
    table.add_column("Tipo", style="cyan", justify="center")
    for entity, label in entities:
        table.add_row(entity, label)

    if entities:
        console.print(table)
    else:
        console.print("[yellow]No se han detectado entidades.[/yellow]")


def _metric_value(metrics: dict[str, float | None], name: str) -> str:
    value = metrics.get(name)
    return "N/A" if value is None else f"{value:.4f}"


def render_eval_tables(metrics: dict[str, float | None]) -> None:
    token_table = Table(title="TOKEN-LEVEL", box=box.SIMPLE_HEAVY)
    token_table.add_column("Métrica", style="bold white")
    token_table.add_column("Valor", justify="right", style="cyan")
    for label, key in (
        ("Accuracy", "token_accuracy"),
        ("Precision", "token_precision"),
        ("Recall", "token_recall"),
        ("F1", "token_f1"),
    ):
        token_table.add_row(label, _metric_value(metrics, key))

    entity_table = Table(title="ENTITY-LEVEL", box=box.SIMPLE_HEAVY)
    entity_table.add_column("Métrica", style="bold white")
    entity_table.add_column("Valor", justify="right", style="cyan")
    for label, key in (
        ("Precision", "entity_precision"),
        ("Recall", "entity_recall"),
        ("F1", "entity_f1"),
    ):
        entity_table.add_row(label, _metric_value(metrics, key))

    per_loc_table = Table(title="PER / LOC", box=box.SIMPLE_HEAVY)
    per_loc_table.add_column("Tipo", style="bold white")
    per_loc_table.add_column("Precision", justify="right", style="cyan")
    per_loc_table.add_column("Recall", justify="right", style="cyan")
    per_loc_table.add_column("F1", justify="right", style="cyan")
    for label, prefix in (("PER", "entity_per"), ("LOC", "entity_loc")):
        per_loc_table.add_row(
            label,
            _metric_value(metrics, f"{prefix}_precision"),
            _metric_value(metrics, f"{prefix}_recall"),
            _metric_value(metrics, f"{prefix}_f1"),
        )

    console.print(token_table)
    console.print(entity_table)
    console.print(per_loc_table)
    console.print(
        "[dim]Las métricas token-level miden etiquetas BIO por token; "
        "las entity-level evalúan entidades completas.[/dim]"
    )


def render_bpe_analysis(analysis: dict[str, object]) -> None:
    table = Table(title="Resumen BPE", box=box.SIMPLE_HEAVY)
    table.add_column("Campo", style="bold white")
    table.add_column("Valor", style="cyan")
    ratio = analysis["chars_per_token"]
    ratio_text = "N/A" if ratio is None else f"{ratio:.2f}"
    table.add_row("Caracteres", str(analysis["n_chars"]))
    table.add_row("Tokens", str(analysis["n_tokens"]))
    table.add_row("Ratio chars/token", ratio_text)

    console.print(
        Panel(str(analysis["text"]), title="Texto original", border_style="cyan")
    )
    console.print(table)
    console.print(Panel(str(analysis["token_ids"]), title="IDs de tokens"))
    console.print(Panel(str(analysis["pieces"]), title="Piezas decodificadas"))
    console.print(Panel(str(analysis["segmentation"]), title="Segmentación"))


def render_command_help() -> None:
    examples = """# Generar texto
uv run fdi-pln-2608-p5 generate --weights checkpoints/p5_causal_2608.pth --prompt "Alice was"

# Detectar entidades
uv run fdi-pln-2608-p5 ner --weights checkpoints/p5_ner_2608.pth --file examples/text.txt

# Evaluar NER
uv run fdi-pln-2608-p5 eval-ner --weights checkpoints/p5_ner_2608.pth --data data/ner/final.conll

# Analizar BPE
uv run fdi-pln-2608-p5 analyze-bpe --weights checkpoints/p5_causal_2608.pth --text "Alice went to Wonderland"

# Ayuda completa
uv run fdi-pln-2608-p5 --help"""
    render_section("Comandos disponibles", "Los comandos directos siguen disponibles.")
    help_console = Console(width=180)
    help_console.print(Syntax(examples, "bash", theme="ansi_dark", word_wrap=False))
