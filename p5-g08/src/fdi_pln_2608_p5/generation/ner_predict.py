import re

import torch

from fdi_pln_2608_p5.checkpoint import load_checkpoint, normalize_config
from fdi_pln_2608_p5.model.ner import NERLLM, NUM_LABELS
from fdi_pln_2608_p5.utils import resolve_device


def simple_tokenize_words(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


def load_ner_model(
    ner_model_path="checkpoints/p5_ner_2608.pth",
    tokenizer_path=None,
    device=None,
):
    device = resolve_device(device)

    checkpoint = load_checkpoint(ner_model_path, map_location=device)
    tokenizer = checkpoint.get("tokenizer")
    if tokenizer is None:
        if tokenizer_path is None:
            raise ValueError(
                "El checkpoint NER no contiene tokenizer; indica --tokenizer-path."
            )
        tokenizer = load_checkpoint(tokenizer_path, map_location=device)

    config = normalize_config(checkpoint["config"])

    model = NERLLM(
        vocab_size=config["vocab_size"],
        max_seq_len=config["context_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        num_labels=checkpoint.get("num_labels", NUM_LABELS),
        expansion=config.get("expansion", 4),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer


def predict_entities_from_text(
    text,
    ner_model_path="checkpoints/p5_ner_2608.pth",
    tokenizer_path=None,
):
    model, tokenizer = load_ner_model(
        ner_model_path=ner_model_path,
        tokenizer_path=tokenizer_path,
    )

    words = simple_tokenize_words(text)
    return model.predict_entities(words, tokenizer)


def predict_entities_from_file(
    file_path,
    ner_model_path="checkpoints/p5_ner_2608.pth",
    tokenizer_path=None,
):
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    return predict_entities_from_text(
        text=text,
        ner_model_path=ner_model_path,
        tokenizer_path=tokenizer_path,
    )
