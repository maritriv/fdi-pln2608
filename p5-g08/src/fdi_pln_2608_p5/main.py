import argparse

from fdi_pln_2608_p5.modules.generate import generate_text
from fdi_pln_2608_p5.modules.ner_predict import (
    predict_entities_from_file,
    predict_entities_from_text,
)
from fdi_pln_2608_p5.modules.train import train_model
from fdi_pln_2608_p5.modules.train_ner import train_ner_model


def main():
    parser = argparse.ArgumentParser(
        description="Práctica 5 PLN - Mini LLM causal y NER"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Entrena el modelo causal para generación de texto.",
    )
    train_parser.add_argument("--resources", default="resources")
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--vocab-size", type=int, default=300)
    train_parser.add_argument("--seq-len", type=int, default=128)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--d-model", type=int, default=128)
    train_parser.add_argument("--n-heads", type=int, default=4)
    train_parser.add_argument("--n-layers", type=int, default=4)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--save-dir", default="checkpoints")
    train_parser.add_argument("--model-name", default="p5_causal_26XX.pth")
    train_parser.add_argument("--resume", action="store_true")

    train_ner_parser = subparsers.add_parser(
        "train-ner",
        help="Entrena el modelo NER usando el backbone del modelo causal.",
    )
    train_ner_parser.add_argument("--ner-data", required=True)
    train_ner_parser.add_argument(
        "--causal-model-path",
        default="checkpoints/p5_causal_26XX.pth",
    )
    train_ner_parser.add_argument(
        "--tokenizer-path",
        default="checkpoints/tokenizer.pt",
    )
    train_ner_parser.add_argument(
        "--save-path",
        default="checkpoints/p5_ner_26XX.pth",
    )
    train_ner_parser.add_argument("--epochs", type=int, default=10)
    train_ner_parser.add_argument("--batch-size", type=int, default=16)
    train_ner_parser.add_argument("--lr", type=float, default=3e-4)

    gen_parser = subparsers.add_parser(
        "generate",
        help="Genera texto a partir de un prompt.",
    )
    gen_parser.add_argument("--prompt", type=str, required=True)
    gen_parser.add_argument("--max-new-tokens", type=int, default=100)
    gen_parser.add_argument("--temperature", type=float, default=0.8)
    gen_parser.add_argument("--top-k", type=int, default=None)
    gen_parser.add_argument(
        "--model-path",
        default="checkpoints/p5_causal_26XX.pth",
    )
    gen_parser.add_argument(
        "--tokenizer-path",
        default="checkpoints/tokenizer.pt",
    )

    ner_parser = subparsers.add_parser(
        "ner",
        help="Encuentra entidades nombradas en un texto o fichero.",
    )
    ner_input = ner_parser.add_mutually_exclusive_group(required=True)
    ner_input.add_argument("--text", type=str)
    ner_input.add_argument("--file", type=str)
    ner_parser.add_argument(
        "--model-path",
        default="checkpoints/p5_ner_26XX.pth",
    )
    ner_parser.add_argument(
        "--tokenizer-path",
        default="checkpoints/tokenizer.pt",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_model(
            resources_path=args.resources,
            epochs=args.epochs,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            learning_rate=args.lr,
            save_dir=args.save_dir,
            model_name=args.model_name,
            resume=args.resume,
        )

    elif args.command == "train-ner":
        train_ner_model(
            ner_data_path=args.ner_data,
            causal_model_path=args.causal_model_path,
            tokenizer_path=args.tokenizer_path,
            save_path=args.save_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )

    elif args.command == "generate":
        text = generate_text(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
        )
        print(text)

    elif args.command == "ner":
        if args.file:
            entities = predict_entities_from_file(
                file_path=args.file,
                ner_model_path=args.model_path,
                tokenizer_path=args.tokenizer_path,
            )
        else:
            entities = predict_entities_from_text(
                text=args.text,
                ner_model_path=args.model_path,
                tokenizer_path=args.tokenizer_path,
            )

        for entity, label in entities:
            print(f"{entity}\t{label}")


if __name__ == "__main__":
    main()
