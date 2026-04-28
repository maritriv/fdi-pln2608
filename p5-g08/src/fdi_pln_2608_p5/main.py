import argparse

from fdi_pln_2608_p5.modules.generate import generate_text
from fdi_pln_2608_p5.modules.train import train_model
from fdi_pln_2608_p5.modules.ner import extract_named_entities


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
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

    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--prompt", type=str, default="Alice")
    gen_parser.add_argument("--max-new-tokens", type=int, default=100)
    gen_parser.add_argument("--temperature", type=float, default=0.8)
    gen_parser.add_argument("--top-k", type=int, default=None)

    ner_parser = subparsers.add_parser("ner")
    ner_parser.add_argument("--text", type=str, required=True)

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
        )

    elif args.command == "generate":
        print(generate_text(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        ))

    elif args.command == "ner":
        print(extract_named_entities(args.text))


if __name__ == "__main__":
    main()