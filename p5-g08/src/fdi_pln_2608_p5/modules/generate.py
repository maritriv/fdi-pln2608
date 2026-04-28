import torch

from fdi_pln_2608_p5.modules.model import MiniLLM


def load_model_and_tokenizer(
    model_path="checkpoints/p5_causal_26XX.pth",
    tokenizer_path="checkpoints/tokenizer.pt",
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(model_path, map_location=device)
    tokenizer = torch.load(tokenizer_path, map_location=device)

    config = checkpoint["config"]

    model = MiniLLM(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        max_seq_len=config["seq_len"],
        dropout=config["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer, device


def generate_text(
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=None,
    model_path="checkpoints/p5_causal_26XX.pth",
    tokenizer_path="checkpoints/tokenizer.pt",
):
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
