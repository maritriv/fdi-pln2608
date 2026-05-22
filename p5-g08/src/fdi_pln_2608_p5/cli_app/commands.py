"""Callbacks Typer y funciones de aplicación del CLI."""

import json
from pathlib import Path
from typing import Annotated

import typer

from fdi_pln_2608_p5.checkpoint import load_checkpoint
from fdi_pln_2608_p5.cli_app.render import console
from fdi_pln_2608_p5.data.prepare_ner_data import convert_merged_to_conll
from fdi_pln_2608_p5.evaluation.evaluate_ner import (
    analyze_bpe,
    evaluate_ner_checkpoint,
)
from fdi_pln_2608_p5.generation.generate import generate_text
from fdi_pln_2608_p5.generation.ner_predict import (
    predict_entities_from_file,
    predict_entities_from_text,
)
from fdi_pln_2608_p5.training.train_causal import train_model
from fdi_pln_2608_p5.training.train_ner import train_ner_model


def train_causal_impl(
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


def train_ner_impl(
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


def prepare_ner_data_impl(input_path: str, output_path: str) -> dict[str, int]:
    return convert_merged_to_conll(input_path=input_path, output_path=output_path)


def generate_impl(
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


def ner_impl(
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


def eval_ner_impl(weights: str, data: str, batch_size: int) -> dict[str, float | None]:
    return evaluate_ner_checkpoint(
        weights=weights,
        data_path=data,
        batch_size=batch_size,
    )


def write_metrics(metrics: dict[str, float | None], out: str) -> Path:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_path


def analyze_bpe_impl(
    weights: str,
    text: str | None,
    file_path: str | None,
) -> dict[str, object]:
    if bool(text) == bool(file_path):
        raise ValueError("Indica exactamente una entrada: --text o --file.")
    return analyze_bpe(weights=weights, text=text, file_path=file_path)


def register_commands(app: typer.Typer) -> None:
    """Registra todos los subcomandos públicos sobre la app Typer."""

    @app.command(
        "train-causal", help="Entrena el modelo causal para generación de texto."
    )
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
        train_causal_impl(
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
        train_causal_impl(
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
        train_ner_impl(
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
        metrics = prepare_ner_data_impl(input_path=input_path, output_path=output)
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
        text = generate_impl(
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
            entities = ner_impl(
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

    @app.command(
        "eval-ner", help="Evalúa un checkpoint NER contra un fichero CoNLL/BIO."
    )
    def eval_ner(
        data: Annotated[str, typer.Option()],
        weights: Annotated[str, typer.Option()] = "checkpoints/p5_ner_2608.pth",
        batch_size: Annotated[int, typer.Option("--batch-size")] = 16,
        out: Annotated[str, typer.Option()] = "reports/ner_metrics_2608.json",
    ) -> None:
        metrics = eval_ner_impl(weights=weights, data=data, batch_size=batch_size)
        console.print("[bold]Métricas NER[/bold]")
        for name, value in metrics.items():
            rendered = "None" if value is None else f"{value:.4f}"
            console.print(f"{name}: {rendered}")
        out_path = write_metrics(metrics, out)
        console.print(f"Métricas guardadas en: {out_path}")

    @app.command(
        "analyze-bpe",
        help="Muestra la segmentación BPE de un texto o fichero.",
    )
    def analyze_bpe_command(
        weights: Annotated[str, typer.Option()] = "checkpoints/p5_causal_2608.pth",
        text: Annotated[str | None, typer.Option("--text")] = None,
        file_path: Annotated[str | None, typer.Option("--file")] = None,
    ) -> None:
        try:
            analysis = analyze_bpe_impl(weights=weights, text=text, file_path=file_path)
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
                generated = generate_impl(
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
