"""CLI de la practica P5: mini LLM causal y NER."""

import argparse
import json
from pathlib import Path

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P5 PLN 2608 - Mini LLM Transformer y NER BIO"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train-causal",
        aliases=["train"],
        help="Entrena el modelo causal para generacion de texto.",
    )
    train_parser.add_argument("--corpus", "--resources", default="resources")
    train_parser.add_argument("--output", default="checkpoints/p5_causal_2608.pth")
    train_parser.add_argument("--epochs", type=int, default=8)
    train_parser.add_argument("--vocab-size", type=int, default=400)
    train_parser.add_argument("--context-size", "--seq-len", type=int, default=128)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--d-model", type=int, default=128)
    train_parser.add_argument("--n-heads", type=int, default=4)
    train_parser.add_argument("--n-layers", type=int, default=4)
    train_parser.add_argument("--expansion", type=int, default=4)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--resume", action="store_true")

    train_ner_parser = subparsers.add_parser(
        "train-ner",
        help="Entrena NER BIO reutilizando el backbone del modelo causal.",
    )
    train_ner_parser.add_argument("--data", "--ner-data", required=True)
    train_ner_parser.add_argument(
        "--causal-weights",
        "--causal-model-path",
        default="checkpoints/p5_causal_2608.pth",
    )
    train_ner_parser.add_argument("--output", default="checkpoints/p5_ner_2608.pth")
    train_ner_parser.add_argument("--tokenizer-path", default=None)
    train_ner_parser.add_argument("--epochs", type=int, default=10)
    train_ner_parser.add_argument("--batch-size", type=int, default=16)
    train_ner_parser.add_argument("--lr", type=float, default=3e-4)
    train_ner_parser.add_argument("--seed", type=int, default=42)

    prepare_ner_parser = subparsers.add_parser(
        "prepare-ner-data",
        help="Convierte el merged.json de la preentrega a CoNLL/BIO.",
    )
    prepare_ner_parser.add_argument(
        "--input",
        required=True,
        help="Ruta al merged.json fusionado de la preentrega.",
    )
    prepare_ner_parser.add_argument(
        "--output",
        default="data/ner/final.conll",
        help="Ruta de salida CoNLL/BIO.",
    )

    gen_parser = subparsers.add_parser(
        "generate",
        help="Genera texto a partir de un prompt.",
    )
    gen_parser.add_argument("--weights", default="checkpoints/p5_causal_2608.pth")
    gen_parser.add_argument("--prompt", type=str, required=True)
    gen_parser.add_argument("--max-new-tokens", type=int, default=100)
    gen_parser.add_argument("--temperature", type=float, default=0.8)
    gen_parser.add_argument("--top-k", type=int, default=None)
    gen_parser.add_argument("--model-path", default=None)
    gen_parser.add_argument("--tokenizer-path", default=None)

    ner_parser = subparsers.add_parser(
        "ner",
        help="Encuentra entidades nombradas en texto o fichero.",
    )
    ner_input = ner_parser.add_mutually_exclusive_group(required=True)
    ner_input.add_argument("--text", type=str)
    ner_input.add_argument("--file", type=str)
    ner_parser.add_argument("--weights", default="checkpoints/p5_ner_2608.pth")
    ner_parser.add_argument("--model-path", default=None)
    ner_parser.add_argument("--tokenizer-path", default=None)

    eval_ner_parser = subparsers.add_parser(
        "eval-ner",
        help="Evalua un checkpoint NER contra un fichero CoNLL/BIO.",
    )
    eval_ner_parser.add_argument("--weights", default="checkpoints/p5_ner_2608.pth")
    eval_ner_parser.add_argument("--data", required=True)
    eval_ner_parser.add_argument("--batch-size", type=int, default=16)
    eval_ner_parser.add_argument("--out", default="reports/ner_metrics_2608.json")

    analyze_parser = subparsers.add_parser(
        "analyze-bpe",
        help="Muestra la segmentacion BPE de un texto o fichero.",
    )
    analyze_parser.add_argument("--weights", default="checkpoints/p5_causal_2608.pth")
    analyze_input = analyze_parser.add_mutually_exclusive_group(required=True)
    analyze_input.add_argument("--text", type=str)
    analyze_input.add_argument("--file", type=str)

    experiment_parser = subparsers.add_parser(
        "experiment-generate",
        help="Genera una tabla markdown con pruebas de temperature y top-k.",
    )
    experiment_parser.add_argument(
        "--weights", default="checkpoints/p5_causal_2608.pth"
    )
    experiment_parser.add_argument("--prompt", required=True)
    experiment_parser.add_argument("--out", default="reports/generation_experiments.md")
    experiment_parser.add_argument("--max-new-tokens", type=int, default=80)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in {"train-causal", "train"}:
        train_model(
            resources_path=args.corpus,
            epochs=args.epochs,
            vocab_size=args.vocab_size,
            context_size=args.context_size,
            batch_size=args.batch_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            expansion=args.expansion,
            dropout=args.dropout,
            learning_rate=args.lr,
            output_path=args.output,
            seed=args.seed,
            resume=args.resume,
        )

    elif args.command == "train-ner":
        train_ner_model(
            ner_data_path=args.data,
            causal_model_path=args.causal_weights,
            tokenizer_path=args.tokenizer_path,
            save_path=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
        )

    elif args.command == "generate":
        text = generate_text(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            weights=args.weights,
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
        )
        print(text)

    elif args.command == "prepare-ner-data":
        metrics = convert_merged_to_conll(
            input_path=args.input,
            output_path=args.output,
        )
        print("Datos NER preparados")
        print("====================")
        for name, value in metrics.items():
            print(f"{name}: {value}")

    elif args.command == "ner":
        ner_model_path = args.model_path or args.weights

        if args.file:
            entities = predict_entities_from_file(
                file_path=args.file,
                ner_model_path=ner_model_path,
                tokenizer_path=args.tokenizer_path,
            )
        else:
            entities = predict_entities_from_text(
                text=args.text,
                ner_model_path=ner_model_path,
                tokenizer_path=args.tokenizer_path,
            )

        for entity, label in entities:
            print(f"{entity}\t{label}")

    elif args.command == "eval-ner":
        metrics = evaluate_ner_checkpoint(
            weights=args.weights,
            data_path=args.data,
            batch_size=args.batch_size,
        )
        print("Metricas NER")
        print("============")
        for name, value in metrics.items():
            rendered = "None" if value is None else f"{value:.4f}"
            print(f"{name}: {rendered}")
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Metricas guardadas en: {out_path}")

    elif args.command == "analyze-bpe":
        analysis = analyze_bpe(
            weights=args.weights,
            text=args.text,
            file_path=args.file,
        )
        print("Analisis BPE")
        print("============")
        print(f"Texto original:\n{analysis['text']}")
        print(f"\nIds de tokens:\n{analysis['token_ids']}")
        print(f"\nPiezas decodificadas:\n{analysis['pieces']}")
        print(f"\nNumero de caracteres: {analysis['n_chars']}")
        print(f"Numero de tokens: {analysis['n_tokens']}")
        ratio = analysis["chars_per_token"]
        ratio_text = "None" if ratio is None else f"{ratio:.2f}"
        print(f"Ratio caracteres/tokens: {ratio_text}")
        print(f"\nSegmentacion:\n{analysis['segmentation']}")

    elif args.command == "experiment-generate":
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = load_checkpoint(args.weights, map_location="cpu")
        config = checkpoint.get("config", {})

        lines = [
            "# Experimentos de generacion",
            "",
            f"- Checkpoint: `{args.weights}`",
            f"- Prompt: `{args.prompt}`",
            f"- Max new tokens: `{args.max_new_tokens}`",
            "",
            "## Configuracion del checkpoint",
            "",
            "| Parametro | Valor |",
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
                generated = generate_text(
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    weights=args.weights,
                )
                generated = generated.replace("\n", "<br>")
                lines.append(f"| {temperature} | {top_k} | {generated} |  |")

        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Experimentos guardados en: {out_path}")


if __name__ == "__main__":
    main()
