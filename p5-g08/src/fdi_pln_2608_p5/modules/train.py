# Entrenamiento del mini LLM

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fdi_pln_2608_p5.modules.data import build_tokenizer_and_dataset
from fdi_pln_2608_p5.modules.model import MiniLLM


def train_model(
    resources_path="resources",
    vocab_size=300,
    seq_len=64,
    batch_size=16,
    d_model=128,
    n_heads=4,
    n_layers=2,
    dropout=0.1,
    learning_rate=3e-4,
    epochs=5,
    device=None,
    save_dir="checkpoints",
):
    """Entrena el mini LLM sobre el corpus indicado."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Usando dispositivo: {device}")

    # 1. Cargamos corpus, tokenizer y dataset
    tokenizer, dataset, text, token_ids = build_tokenizer_and_dataset(
        resources_path=resources_path,
        vocab_size=vocab_size,
        seq_len=seq_len,
    )

    print(f"Corpus cargado: {len(text)} caracteres")
    print(f"Vocabulario aprendido: {len(tokenizer.vocab)} tokens")
    print(f"Corpus tokenizado: {len(token_ids)} tokens")
    print(f"Ejemplos de entrenamiento: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Creamos modelo
    model = MiniLLM(
        vocab_size=len(tokenizer.vocab),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=seq_len,
        dropout=dropout,
    ).to(device)

    # 3. Optimizador
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 4. Bucle de entrenamiento
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for step, (x, y) in enumerate(dataloader, start=1):
            x = x.to(device)
            y = y.to(device)

            # Forward:
            # logits -> (batch_size, seq_len, vocab_size)
            logits = model(x, causal=True)

            # CrossEntropy espera:
            # input:  (N, C)
            # target: (N,)
            # así que aplanamos batch y tiempo
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0:
                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"Step {step}/{len(dataloader)} | "
                    f"Loss {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{epochs} completada | Loss media: {avg_loss:.4f}")

    # 5. Guardamos checkpoint
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    checkpoint_file = save_path / "mini_llm.pt"
    tokenizer_file = save_path / "tokenizer.pt"

    torch.save(model.state_dict(), checkpoint_file)
    torch.save(tokenizer, tokenizer_file)

    print(f"Modelo guardado en: {checkpoint_file}")
    print(f"Tokenizer guardado en: {tokenizer_file}")

    return model, tokenizer


if __name__ == "__main__":
    train_model()
