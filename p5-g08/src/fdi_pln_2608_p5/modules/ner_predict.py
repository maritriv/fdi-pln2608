import re

import torch

from fdi_pln_2608_p5.modules.ner import NERLLM, NUM_LABELS


def simple_tokenize_words(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


def load_ner_model(
    ner_model_path="checkpoints/p5_ner_26XX.pth",
    tokenizer_path="checkpoints/tokenizer.pt",
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = torch.load(tokenizer_path, map_location=device)
    checkpoint = torch.load(ner_model_path, map_location=device)

    config = checkpoint["config"]

    model = NERLLM(
        vocab_size=config["vocab_size"],
        max_seq_len=config["seq_len"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        num_labels=checkpoint.get("num_labels", NUM_LABELS),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer


def predict_entities_from_text(
    text,
    ner_model_path="checkpoints/p5_ner_26XX.pth",
    tokenizer_path="checkpoints/tokenizer.pt",
):
    model, tokenizer = load_ner_model(
        ner_model_path=ner_model_path,
        tokenizer_path=tokenizer_path,
    )

    words = simple_tokenize_words(text)
    return model.predict_entities(words, tokenizer)


def predict_entities_from_file(
    file_path,
    ner_model_path="checkpoints/p5_ner_26XX.pth",
    tokenizer_path="checkpoints/tokenizer.pt",
):
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    return predict_entities_from_text(
        text=text,
        ner_model_path=ner_model_path,
        tokenizer_path=tokenizer_path,
    )
