# Entrenamiento del mini LLM

from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from fdi_pln_2608_p5.modules.data import build_tokenizer_and_dataset
from fdi_pln_2608_p5.modules.model import MiniLLM


def run_epoch(model, dataloader, device, optimizer=None):
    """Ejecuta una época de entrenamiento o validación."""

    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            if is_train:
                optimizer.zero_grad()

            _, loss = model(x, targets=y, causal=True)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(
    resources_path="resources",
    vocab_size=300,
    seq_len=128,
    batch_size=64,
    d_model=128,
    n_heads=4,
    n_layers=4,
    dropout=0.1,
    learning_rate=3e-4,
    epochs=5,
    device=None,
    save_dir="checkpoints",
    resume=False,
):
    """Entrena el mini LLM sobre el corpus indicado."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    checkpoint_file = save_path / "mini_llm.pt"
    tokenizer_file = save_path / "tokenizer.pt"

    print(f"Usando dispositivo: {device}")

    tokenizer, train_ds, val_ds, text = build_tokenizer_and_dataset(
        resources_path=resources_path,
        vocab_size=vocab_size,
        seq_len=seq_len,
    )

    print(f"Corpus cargado: {len(text)} caracteres")
    print(f"Vocabulario aprendido: {len(tokenizer.vocab)} tokens")
    print(f"Ejemplos de entrenamiento: {len(train_ds)}")
    print(f"Ejemplos de validación: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    model = MiniLLM(
        vocab_size=len(tokenizer.vocab),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=seq_len,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    start_epoch = 1

    if resume and checkpoint_file.exists():
        checkpoint = torch.load(checkpoint_file, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint.get("epoch", 0) + 1

        print(f"Checkpoint cargado desde: {checkpoint_file}")
        print(f"Continuando desde epoch {start_epoch}")

    t0 = time.time()

    for epoch in range(start_epoch, epochs + 1):
        train_loss = run_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
        )

        val_loss = run_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            optimizer=None,
        )

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"tiempo={elapsed:.1f}s"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "vocab_size": len(tokenizer.vocab),
                "seq_len": seq_len,
                "batch_size": batch_size,
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "dropout": dropout,
                "learning_rate": learning_rate,
            },
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        torch.save(checkpoint, checkpoint_file)
        torch.save(tokenizer, tokenizer_file)

    print(f"Modelo guardado en: {checkpoint_file}")
    print(f"Tokenizer guardado en: {tokenizer_file}")

    return model, tokenizer


if __name__ == "__main__":
    train_model()