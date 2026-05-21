"""CLI de la práctica P5: mini LLM causal y NER."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from fdi_pln_2608_p5.checkpoint import load_checkpoint
from fdi_pln_2608_p5.evaluate import analyze_bpe, evaluate_ner_checkpoint
from fdi_pln_2608_p5.modules.generate import generate_text
from fdi_pln_2608_p5.modules.ner_predict import (
    predict_entities_from_file,
    predict_entities_from_text,
)
from fdi_pln_2608_p5.modules.train import train_model
from fdi_pln_2608_p5.modules.train_ner import train_ner_model
from fdi_pln_2608_p5.prepare_ner_data import convert_merged_to_conll


console = Console()

app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help=("P5 PLN 2608: mini LLM Transformer causal, tokenización BPE y NER BIO."),
    invoke_without_command=True,
    no_args_is_help=False,
    rich_markup_mode="rich",
)


def _train_causal_impl(
    corpus: str,
    output: str,
    epochs: int,
    vocab_size: int,
    context_size: int,
    batch_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    expansion: int,
    dropout: float,
    lr: float,
    seed: int,
    resume: bool,
) -> None:
    train_model(
        resources_path=corpus,
        epochs=epochs,
        vocab_size=vocab_size,
        context_size=context_size,
        batch_size=batch_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        expansion=expansion,
        dropout=dropout,
        learning_rate=lr,
        output_path=output,
        seed=seed,
        resume=resume,
    )


def _train_ner_impl(
    data: str,
    causal_weights: str,
    output: str,
    tokenizer_path: str | None,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> None:
    train_ner_model(
        ner_data_path=data,
        causal_model_path=causal_weights,
        tokenizer_path=tokenizer_path,
        save_path=output,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        seed=seed,
    )


def _prepare_ner_data_impl(input_path: str, output_path: str) -> dict[str, int]:
    return convert_merged_to_conll(input_path=input_path, output_path=output_path)


def _generate_impl(
    weights: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    model_path: str | None = None,
    tokenizer_path: str | None = None,
) -> str:
    return generate_text(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        weights=weights,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )


def _ner_impl(
    weights: str,
    text: str | None,
    file_path: str | None,
    model_path: str | None = None,
    tokenizer_path: str | None = None,
) -> list[tuple[str, str]]:
    if bool(text) == bool(file_path):
        raise ValueError("Indica exactamente una entrada: --text o --file.")

    ner_model_path = model_path or weights
    if file_path:
        return predict_entities_from_file(
            file_path=file_path,
            ner_model_path=ner_model_path,
            tokenizer_path=tokenizer_path,
        )
    return predict_entities_from_text(
        text=text or "",
        ner_model_path=ner_model_path,
        tokenizer_path=tokenizer_path,
    )


def _eval_ner_impl(weights: str, data: str, batch_size: int) -> dict[str, float | None]:
    return evaluate_ner_checkpoint(
        weights=weights,
        data_path=data,
        batch_size=batch_size,
    )


def _write_metrics(metrics: dict[str, float | None], out: str) -> Path:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_path


def _analyze_bpe_impl(
    weights: str,
    text: str | None,
    file_path: str | None,
) -> dict[str, object]:
    if bool(text) == bool(file_path):
        raise ValueError("Indica exactamente una entrada: --text o --file.")
    return analyze_bpe(weights=weights, text=text, file_path=file_path)


def _render_header() -> None:
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


def _render_section(title: str, description: str | None = None) -> None:
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))
    if description:
        console.print(f"[dim]{description}[/dim]\n")


def _render_success(message: str) -> None:
    console.print(f"[green]OK[/green] {message}")


def _render_error(message: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {message}")


def _pause() -> None:
    Prompt.ask(
        "\n[dim]Pulsa Enter para volver al menú[/dim]", default="", show_default=False
    )


def _ask_with_default(label: str, default: str) -> str:
    value = Prompt.ask(
        f"{label} [cyan]\\[{escape(default)}][/cyan]",
        default="",
        show_default=False,
    )
    return value.strip() or default


def _ask_int_with_default(label: str, default: int) -> int:
    while True:
        value = _ask_with_default(label, str(default))
        try:
            return int(value)
        except ValueError:
            _render_error("Introduce un número entero válido.")


def _ask_float_with_default(label: str, default: float) -> float:
    while True:
        value = _ask_with_default(label, str(default))
        try:
            return float(value)
        except ValueError:
            _render_error("Introduce un número decimal válido.")


def _path_exists(path: str) -> bool:
    if Path(path).exists():
        return True
    _render_error(f"No se ha encontrado el fichero indicado: {path}")
    return False


def _render_menu() -> None:
    _render_header()
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


def _render_entities_table(entities: list[tuple[str, str]]) -> None:
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


def _render_eval_tables(metrics: dict[str, float | None]) -> None:
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


def _render_bpe_analysis(analysis: dict[str, object]) -> None:
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


def _render_command_help() -> None:
    examples = """# Generar texto
uv run fdi-pln-2608-p5 generate \\
  --weights checkpoints/p5_causal_2608.pth \\
  --prompt "Alice was"

# Detectar entidades
uv run fdi-pln-2608-p5 ner \\
  --weights checkpoints/p5_ner_2608.pth \\
  --file examples/text.txt

# Evaluar NER
uv run fdi-pln-2608-p5 eval-ner \\
  --weights checkpoints/p5_ner_2608.pth \\
  --data data/ner/final.conll

# Analizar BPE
uv run fdi-pln-2608-p5 analyze-bpe \\
  --weights checkpoints/p5_causal_2608.pth \\
  --text "Alice went to Wonderland"

# Ayuda completa
uv run fdi-pln-2608-p5 --help"""
    _render_section("Comandos disponibles", "Los comandos directos siguen disponibles.")
    console.print(Syntax(examples, "bash", theme="ansi_dark", word_wrap=True))


def _interactive_generate() -> None:
    _render_section(
        "Generar texto",
        "Carga el modelo causal y continua un prompt token a token.",
    )
    console.print("[dim]Pulsa Enter para usar el valor por defecto.[/dim]\n")
    weights = _ask_with_default("Checkpoint", "checkpoints/p5_causal_2608.pth")
    if not _path_exists(weights):
        return
    prompt = _ask_with_default("Prompt", "Alice was")
    max_new_tokens = _ask_int_with_default("Max new tokens", 80)
    top_k = _ask_int_with_default("Top-k", 20)
    temperature = _ask_float_with_default("Temperature", 1.0)

    with console.status("[bold cyan]Cargando modelo y generando texto...[/bold cyan]"):
        text = _generate_impl(
            weights=weights,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    console.print(Panel(text, title="Resultado", border_style="green"))


def _interactive_ner() -> None:
    _render_section(
        "Detectar entidades NER",
        "Lee un fichero de texto y lista entidades PER/LOC detectadas.",
    )
    console.print("[dim]Pulsa Enter para usar el valor por defecto.[/dim]\n")
    weights = _ask_with_default("Checkpoint", "checkpoints/p5_ner_2608.pth")
    file_path = _ask_with_default("Fichero", "examples/text.txt")
    if not _path_exists(weights) or not _path_exists(file_path):
        return

    with console.status("[bold cyan]Cargando modelo NER...[/bold cyan]"):
        entities = _ner_impl(weights=weights, text=None, file_path=file_path)
    _render_entities_table(entities)


def _interactive_eval_ner() -> None:
    _render_section(
        "Evaluar modelo NER",
        "Calcula métricas token-level y entity-level sobre un dataset CoNLL.",
    )
    console.print("[dim]Pulsa Enter para usar el valor por defecto.[/dim]\n")
    weights = _ask_with_default("Checkpoint", "checkpoints/p5_ner_2608.pth")
    data = _ask_with_default("Dataset CoNLL", "data/ner/final.conll")
    if not _path_exists(weights) or not _path_exists(data):
        return

    with console.status("[bold cyan]Evaluando checkpoint NER...[/bold cyan]"):
        metrics = _eval_ner_impl(weights=weights, data=data, batch_size=16)
        out_path = _write_metrics(metrics, "reports/ner_metrics_2608.json")
    _render_eval_tables(metrics)
    _render_success(f"Métricas guardadas en {out_path}")


def _interactive_analyze_bpe() -> None:
    _render_section(
        "Analizar tokenización BPE",
        "Muestra cómo el tokenizador BPE segmenta un texto de entrada.",
    )
    console.print("[dim]Pulsa Enter para usar el valor por defecto.[/dim]\n")
    weights = _ask_with_default("Checkpoint", "checkpoints/p5_causal_2608.pth")
    if not _path_exists(weights):
        return
    text = _ask_with_default("Texto", "Alice went to Wonderland")

    with console.status("[bold cyan]Analizando tokenización...[/bold cyan]"):
        analysis = _analyze_bpe_impl(weights=weights, text=text, file_path=None)
    _render_bpe_analysis(analysis)


def run_interactive_menu() -> None:
    while True:
        console.clear()
        _render_menu()
        choice = Prompt.ask("Selección").strip()

        if choice == "0":
            console.print("\n[green]Hasta luego.[/green]")
            return

        try:
            if choice == "1":
                _interactive_generate()
            elif choice == "2":
                _interactive_ner()
            elif choice == "3":
                _interactive_eval_ner()
            elif choice == "4":
                _interactive_analyze_bpe()
            elif choice == "5":
                _render_command_help()
            else:
                _render_error("Opción no válida. Elige un número entre 0 y 5.")
        except KeyboardInterrupt:
            console.print("\n[yellow]Operación cancelada.[/yellow]")
        except Exception as exc:
            _render_error(str(exc))

        _pause()


@app.callback()
def _main_callback(ctx: typer.Context) -> None:
    """Abre el menú interactivo si no se indica un subcomando."""

    if ctx.invoked_subcommand is None:
        run_interactive_menu()
        raise typer.Exit()


def _train_options(
    corpus: Annotated[str, typer.Option("--corpus", "--resources")] = "resources",
    output: Annotated[str, typer.Option()] = "checkpoints/p5_causal_2608.pth",
    epochs: Annotated[int, typer.Option()] = 8,
    vocab_size: Annotated[int, typer.Option("--vocab-size")] = 400,
    context_size: Annotated[int, typer.Option("--context-size", "--seq-len")] = 128,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 32,
    d_model: Annotated[int, typer.Option("--d-model")] = 128,
    n_heads: Annotated[int, typer.Option("--n-heads")] = 4,
    n_layers: Annotated[int, typer.Option("--n-layers")] = 4,
    expansion: Annotated[int, typer.Option()] = 4,
    dropout: Annotated[float, typer.Option()] = 0.1,
    lr: Annotated[float, typer.Option()] = 3e-4,
    seed: Annotated[int, typer.Option()] = 42,
    resume: Annotated[bool, typer.Option()] = False,
) -> None:
    _train_causal_impl(
        corpus=corpus,
        output=output,
        epochs=epochs,
        vocab_size=vocab_size,
        context_size=context_size,
        batch_size=batch_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        expansion=expansion,
        dropout=dropout,
        lr=lr,
        seed=seed,
        resume=resume,
    )


@app.command("train-causal", help="Entrena el modelo causal para generación de texto.")
def train_causal(
    corpus: Annotated[str, typer.Option("--corpus", "--resources")] = "resources",
    output: Annotated[str, typer.Option()] = "checkpoints/p5_causal_2608.pth",
    epochs: Annotated[int, typer.Option()] = 8,
    vocab_size: Annotated[int, typer.Option("--vocab-size")] = 400,
    context_size: Annotated[int, typer.Option("--context-size", "--seq-len")] = 128,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 32,
    d_model: Annotated[int, typer.Option("--d-model")] = 128,
    n_heads: Annotated[int, typer.Option("--n-heads")] = 4,
    n_layers: Annotated[int, typer.Option("--n-layers")] = 4,
    expansion: Annotated[int, typer.Option()] = 4,
    dropout: Annotated[float, typer.Option()] = 0.1,
    lr: Annotated[float, typer.Option()] = 3e-4,
    seed: Annotated[int, typer.Option()] = 42,
    resume: Annotated[bool, typer.Option()] = False,
) -> None:
    _train_options(
        corpus=corpus,
        output=output,
        epochs=epochs,
        vocab_size=vocab_size,
        context_size=context_size,
        batch_size=batch_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        expansion=expansion,
        dropout=dropout,
        lr=lr,
        seed=seed,
        resume=resume,
    )


@app.command("train", help="Alias de train-causal.")
def train_alias(
    corpus: Annotated[str, typer.Option("--corpus", "--resources")] = "resources",
    output: Annotated[str, typer.Option()] = "checkpoints/p5_causal_2608.pth",
    epochs: Annotated[int, typer.Option()] = 8,
    vocab_size: Annotated[int, typer.Option("--vocab-size")] = 400,
    context_size: Annotated[int, typer.Option("--context-size", "--seq-len")] = 128,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 32,
    d_model: Annotated[int, typer.Option("--d-model")] = 128,
    n_heads: Annotated[int, typer.Option("--n-heads")] = 4,
    n_layers: Annotated[int, typer.Option("--n-layers")] = 4,
    expansion: Annotated[int, typer.Option()] = 4,
    dropout: Annotated[float, typer.Option()] = 0.1,
    lr: Annotated[float, typer.Option()] = 3e-4,
    seed: Annotated[int, typer.Option()] = 42,
    resume: Annotated[bool, typer.Option()] = False,
) -> None:
    train_causal(
        corpus=corpus,
        output=output,
        epochs=epochs,
        vocab_size=vocab_size,
        context_size=context_size,
        batch_size=batch_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        expansion=expansion,
        dropout=dropout,
        lr=lr,
        seed=seed,
        resume=resume,
    )


@app.command("train-ner", help="Entrena NER BIO reutilizando el backbone causal.")
def train_ner(
    data: Annotated[str, typer.Option("--data", "--ner-data")],
    causal_weights: Annotated[
        str,
        typer.Option("--causal-weights", "--causal-model-path"),
    ] = "checkpoints/p5_causal_2608.pth",
    output: Annotated[str, typer.Option()] = "checkpoints/p5_ner_2608.pth",
    tokenizer_path: Annotated[str | None, typer.Option("--tokenizer-path")] = None,
    epochs: Annotated[int, typer.Option()] = 10,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 16,
    lr: Annotated[float, typer.Option()] = 3e-4,
    seed: Annotated[int, typer.Option()] = 42,
) -> None:
    _train_ner_impl(
        data=data,
        causal_weights=causal_weights,
        output=output,
        tokenizer_path=tokenizer_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
    )


@app.command("prepare-ner-data", help="Convierte merged.json a formato CoNLL/BIO.")
def prepare_ner_data(
    input_path: Annotated[
        str,
        typer.Option("--input", help="Ruta al merged.json fusionado."),
    ],
    output: Annotated[str, typer.Option()] = "data/ner/final.conll",
) -> None:
    metrics = _prepare_ner_data_impl(input_path=input_path, output_path=output)
    console.print("[bold]Datos NER preparados[/bold]")
    for name, value in metrics.items():
        console.print(f"{name}: {value}")


@app.command("generate", help="Genera texto a partir de un prompt.")
def generate(
    prompt: Annotated[str, typer.Option()],
    weights: Annotated[str, typer.Option()] = "checkpoints/p5_causal_2608.pth",
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens")] = 100,
    temperature: Annotated[float, typer.Option()] = 0.8,
    top_k: Annotated[int | None, typer.Option("--top-k")] = None,
    model_path: Annotated[str | None, typer.Option("--model-path")] = None,
    tokenizer_path: Annotated[str | None, typer.Option("--tokenizer-path")] = None,
) -> None:
    text = _generate_impl(
        weights=weights,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )
    console.print(text)


@app.command("ner", help="Encuentra entidades nombradas en texto o fichero.")
def ner(
    text: Annotated[str | None, typer.Option("--text")] = None,
    file_path: Annotated[str | None, typer.Option("--file")] = None,
    weights: Annotated[str, typer.Option()] = "checkpoints/p5_ner_2608.pth",
    model_path: Annotated[str | None, typer.Option("--model-path")] = None,
    tokenizer_path: Annotated[str | None, typer.Option("--tokenizer-path")] = None,
) -> None:
    try:
        entities = _ner_impl(
            weights=weights,
            text=text,
            file_path=file_path,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    for entity, label in entities:
        console.print(f"{entity}\t{label}")


@app.command("eval-ner", help="Evalúa un checkpoint NER contra un fichero CoNLL/BIO.")
def eval_ner(
    data: Annotated[str, typer.Option()],
    weights: Annotated[str, typer.Option()] = "checkpoints/p5_ner_2608.pth",
    batch_size: Annotated[int, typer.Option("--batch-size")] = 16,
    out: Annotated[str, typer.Option()] = "reports/ner_metrics_2608.json",
) -> None:
    metrics = _eval_ner_impl(weights=weights, data=data, batch_size=batch_size)
    console.print("[bold]Métricas NER[/bold]")
    for name, value in metrics.items():
        rendered = "None" if value is None else f"{value:.4f}"
        console.print(f"{name}: {rendered}")
    out_path = _write_metrics(metrics, out)
    console.print(f"Métricas guardadas en: {out_path}")


@app.command("analyze-bpe", help="Muestra la segmentación BPE de un texto o fichero.")
def analyze_bpe_command(
    weights: Annotated[str, typer.Option()] = "checkpoints/p5_causal_2608.pth",
    text: Annotated[str | None, typer.Option("--text")] = None,
    file_path: Annotated[str | None, typer.Option("--file")] = None,
) -> None:
    try:
        analysis = _analyze_bpe_impl(weights=weights, text=text, file_path=file_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    console.print("[bold]Análisis BPE[/bold]")
    console.print(f"Texto original:\n{analysis['text']}")
    console.print(f"\nIds de tokens:\n{analysis['token_ids']}")
    console.print(f"\nPiezas decodificadas:\n{analysis['pieces']}")
    console.print(f"\nNúmero de caracteres: {analysis['n_chars']}")
    console.print(f"Número de tokens: {analysis['n_tokens']}")
    ratio = analysis["chars_per_token"]
    ratio_text = "None" if ratio is None else f"{ratio:.2f}"
    console.print(f"Ratio caracteres/tokens: {ratio_text}")
    console.print(f"\nSegmentación:\n{analysis['segmentation']}")


@app.command(
    "experiment-generate",
    help="Genera una tabla markdown con pruebas de temperature y top-k.",
)
def experiment_generate(
    prompt: Annotated[str, typer.Option()],
    weights: Annotated[str, typer.Option()] = "checkpoints/p5_causal_2608.pth",
    out: Annotated[str, typer.Option()] = "reports/generation_experiments.md",
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens")] = 80,
) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(weights, map_location="cpu")
    config = checkpoint.get("config", {})

    lines = [
        "# Experimentos de generación",
        "",
        f"- Checkpoint: `{weights}`",
        f"- Prompt: `{prompt}`",
        f"- Max new tokens: `{max_new_tokens}`",
        "",
        "## Configuración del checkpoint",
        "",
        "| Parámetro | Valor |",
        "| --- | --- |",
    ]
    for key, value in sorted(config.items()):
        lines.append(f"| `{key}` | `{value}` |")

    lines.extend(
        [
            "",
            "## Generaciones",
            "",
            "| Temperature | Top-k | Texto generado | Observaciones |",
            "| ---: | ---: | --- | --- |",
        ]
    )

    for temperature in (0.5, 0.8, 1.2):
        for top_k in (10, 20, 50):
            generated = _generate_impl(
                weights=weights,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            generated = generated.replace("\n", "<br>")
            lines.append(f"| {temperature} | {top_k} | {generated} |  |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"Experimentos guardados en: {out_path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
