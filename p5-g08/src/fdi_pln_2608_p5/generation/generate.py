import torch

from fdi_pln_2608_p5.checkpoint import load_checkpoint, normalize_config
from fdi_pln_2608_p5.model.transformer import MiniLLM
from fdi_pln_2608_p5.utils import resolve_device


def load_model_and_tokenizer(
    model_path="checkpoints/p5_causal_2608.pth",
    tokenizer_path=None,
    device=None,
):
    device = resolve_device(device)

    checkpoint = load_checkpoint(model_path, map_location=device)
    tokenizer = checkpoint.get("tokenizer")

    if tokenizer is None:
        if tokenizer_path is None:
            raise ValueError(
                "El checkpoint no contiene tokenizer; indica --tokenizer-path."
            )
        tokenizer = load_checkpoint(tokenizer_path, map_location=device)

    config = normalize_config(checkpoint["config"])

    model = MiniLLM(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        max_seq_len=config["context_size"],
        dropout=config["dropout"],
        expansion=config.get("expansion", 4),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer, device


def generate_text(
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=None,
    weights="checkpoints/p5_causal_2608.pth",
    model_path=None,
    tokenizer_path=None,
):
    model_path = model_path or weights
    model, tokenizer, _ = load_model_and_tokenizer(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )

    prompt_ids = tokenizer.encode(prompt)

    generated_ids = model.generate(
        prompt_ids,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    return prompt + tokenizer.decode(generated_ids)
