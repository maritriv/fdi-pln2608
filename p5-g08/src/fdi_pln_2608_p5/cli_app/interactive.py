"""Menú interactivo de la práctica P5."""

from rich.panel import Panel
from rich.prompt import Prompt

from fdi_pln_2608_p5.cli_app.commands import (
    analyze_bpe_impl,
    eval_ner_impl,
    generate_impl,
    ner_impl,
    write_metrics,
)
from fdi_pln_2608_p5.cli_app.render import (
    ask_float_with_default,
    ask_int_with_default,
    ask_with_default,
    console,
    path_exists,
    pause,
    render_bpe_analysis,
    render_command_help,
    render_entities_table,
    render_error,
    render_eval_tables,
    render_menu,
    render_section,
    render_success,
)


def _interactive_generate() -> None:
    render_section(
        "Generar texto",
        "Carga el modelo causal y continua un prompt token a token.",
    )
    console.print("[dim]Pulsa Enter para usar el valor por defecto.[/dim]\n")
    weights = ask_with_default("Checkpoint", "checkpoints/p5_causal_2608.pth")
    if not path_exists(weights):
        return
    prompt = ask_with_default("Prompt", "Alice was")
    max_new_tokens = ask_int_with_default("Max new tokens", 80)
    top_k = ask_int_with_default("Top-k", 20)
    temperature = ask_float_with_default("Temperature", 1.0)

    with console.status("[bold cyan]Cargando modelo y generando texto...[/bold cyan]"):
        text = generate_impl(
            weights=weights,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    console.print(Panel(text, title="Resultado", border_style="green"))


def _interactive_ner() -> None:
    render_section(
        "Detectar entidades NER",
        "Lee un fichero de texto y lista entidades PER/LOC detectadas.",
    )
    console.print("[dim]Pulsa Enter para usar el valor por defecto.[/dim]\n")
    weights = ask_with_default("Checkpoint", "checkpoints/p5_ner_2608.pth")
    file_path = ask_with_default("Fichero", "examples/text.txt")
    if not path_exists(weights) or not path_exists(file_path):
        return

    with console.status("[bold cyan]Cargando modelo NER...[/bold cyan]"):
        entities = ner_impl(weights=weights, text=None, file_path=file_path)
    render_entities_table(entities)


def _interactive_eval_ner() -> None:
    render_section(
        "Evaluar modelo NER",
        "Calcula métricas token-level y entity-level sobre un dataset CoNLL.",
    )
    console.print("[dim]Pulsa Enter para usar el valor por defecto.[/dim]\n")
    weights = ask_with_default("Checkpoint", "checkpoints/p5_ner_2608.pth")
    data = ask_with_default("Dataset CoNLL", "data/ner/final.conll")
    if not path_exists(weights) or not path_exists(data):
        return

    with console.status("[bold cyan]Evaluando checkpoint NER...[/bold cyan]"):
        metrics = eval_ner_impl(weights=weights, data=data, batch_size=16)
        out_path = write_metrics(metrics, "reports/ner_metrics_2608.json")
    render_eval_tables(metrics)
    render_success(f"Métricas guardadas en {out_path}")


def _interactive_analyze_bpe() -> None:
    render_section(
        "Analizar tokenización BPE",
        "Muestra cómo el tokenizador BPE segmenta un texto de entrada.",
    )
    console.print("[dim]Pulsa Enter para usar el valor por defecto.[/dim]\n")
    weights = ask_with_default("Checkpoint", "checkpoints/p5_causal_2608.pth")
    if not path_exists(weights):
        return
    text = ask_with_default("Texto", "Alice went to Wonderland")

    with console.status("[bold cyan]Analizando tokenización...[/bold cyan]"):
        analysis = analyze_bpe_impl(weights=weights, text=text, file_path=None)
    render_bpe_analysis(analysis)


def run_interactive_menu() -> None:
    while True:
        console.clear()
        render_menu()
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
                render_command_help()
            else:
                render_error("Opción no válida. Elige un número entre 0 y 5.")
        except KeyboardInterrupt:
            console.print("\n[yellow]Operación cancelada.[/yellow]")
        except Exception as exc:
            render_error(str(exc))

        pause()
