import argparse

from fdi_pln_2608_p5.modules.generate import generate_text
from fdi_pln_2608_p5.modules.train import train_model


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--resources", default="resources")
    train_parser.add_argument("--epochs", type=int, default=5)

    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--prompt", type=str, default="alice")
    gen_parser.add_argument("--max-new-tokens", type=int, default=100)
    gen_parser.add_argument("--temperature", type=float, default=0.8)

    args = parser.parse_args()

    if args.command == "train":
        train_model(resources_path=args.resources, epochs=args.epochs)
    elif args.command == "generate":
        text = generate_text(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(text)


if __name__ == "__main__":
    main()
