# Generación de texto con el mini LLM

import torch
import torch.nn.functional as F

from fdi_pln_2608_p5.modules.model import MiniLLM


def load_model_and_tokenizer(
    checkpoint_path="checkpoints/mini_llm.pt",
    tokenizer_path="checkpoints/tokenizer.pt",
    d_model=128,
    n_heads=4,
    n_layers=2,
    max_seq_len=64,
    dropout=0.1,
    device=None,
):
    """Carga modelo y tokenizador guardados en disco."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = torch.load(tokenizer_path, map_location=device)

    model = MiniLLM(
        vocab_size=len(tokenizer.vocab),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    return model, tokenizer, device


def sample_next_token(logits, temperature=1.0):
    """Muestrea el siguiente token a partir de logits.

    temperature:
    - < 1.0 => más conservador
    - = 1.0 => normal
    - > 1.0 => más aleatorio
    """

    if temperature <= 0:
        raise ValueError("temperature debe ser > 0")

    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


@torch.no_grad()
def generate_text(
    prompt,
    checkpoint_path="checkpoints/mini_llm.pt",
    tokenizer_path="checkpoints/tokenizer.pt",
    max_new_tokens=100,
    temperature=1.0,
    d_model=128,
    n_heads=4,
    n_layers=2,
    max_seq_len=64,
    dropout=0.1,
    device=None,
):
    """Genera texto autoregresivamente a partir de un prompt."""

    model, tokenizer, device = load_model_and_tokenizer(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        dropout=dropout,
        device=device,
    )

    # Codificamos el prompt
    token_ids = tokenizer.encode(prompt)

    if not token_ids:
        raise ValueError("El prompt no puede quedar vacío tras tokenizarse")

    # Convertimos a tensor con dimensión batch
    x = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        # Si el contexto es más largo que la ventana del modelo,
        # usamos solo los últimos max_seq_len tokens
        x_cond = x[:, -max_seq_len:]

        # Forward del modelo
        logits = model(x_cond, causal=True)

        # Nos quedamos con los logits de la última posición
        last_logits = logits[:, -1, :]

        # Muestreamos el siguiente token
        next_token = sample_next_token(last_logits, temperature=temperature)

        # Lo concatenamos a la secuencia actual
        x = torch.cat([x, next_token], dim=1)

    generated_ids = x[0].tolist()
    generated_text = tokenizer.decode(generated_ids)

    return generated_text


if __name__ == "__main__":
    text = generate_text(
        prompt="alice was",
        checkpoint_path="checkpoints/mini_llm.pt",
        tokenizer_path="checkpoints/tokenizer.pt",
        max_new_tokens=100,
        temperature=0.8,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_seq_len=64,
        dropout=0.1,
    )

    print(text)
