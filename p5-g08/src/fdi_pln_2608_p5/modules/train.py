from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from fdi_pln_2608_p5.checkpoint import (
    load_checkpoint,
    normalize_config,
    save_checkpoint,
)
from fdi_pln_2608_p5.modules.data import build_tokenizer_and_dataset
from fdi_pln_2608_p5.modules.model import MiniLLM
from fdi_pln_2608_p5.utils import resolve_device, set_seed


def run_epoch(model, dataloader, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

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
    vocab_size=400,
    seq_len=None,
    context_size=128,
    batch_size=32,
    d_model=128,
    n_heads=4,
    n_layers=4,
    expansion=4,
    dropout=0.1,
    learning_rate=3e-4,
    epochs=8,
    seed=42,
    device=None,
    save_dir="checkpoints",
    model_name="p5_causal_2608.pth",
    output_path=None,
    tokenizer_name="tokenizer.pt",
    resume=False,
):
    set_seed(seed)
    device = resolve_device(device)
    context_size = seq_len if seq_len is not None else context_size

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    checkpoint_file = Path(output_path) if output_path else save_path / model_name
    tokenizer_file = save_path / tokenizer_name

    print(f"Usando dispositivo: {device}")

    tokenizer, train_ds, val_ds, text = build_tokenizer_and_dataset(
        resources_path=resources_path,
        vocab_size=vocab_size,
        seq_len=context_size,
    )

    print(f"Corpus cargado: {len(text)} caracteres")
    print(f"Vocabulario aprendido: {len(tokenizer.vocab)} tokens")
    print(f"Ejemplos de entrenamiento: {len(train_ds)}")
    print(f"Ejemplos de validación: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MiniLLM(
        vocab_size=len(tokenizer.vocab),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=context_size,
        dropout=dropout,
        expansion=expansion,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start_epoch = 1

    if resume and checkpoint_file.exists():
        checkpoint = load_checkpoint(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Checkpoint cargado desde: {checkpoint_file}")

    t0 = time.time()

    for epoch in range(start_epoch, epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"tiempo={elapsed:.1f}s"
        )

        config = normalize_config(
            {
                "task": "causal_lm",
                "vocab_size": len(tokenizer.vocab),
                "context_size": context_size,
                "batch_size": batch_size,
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "expansion": expansion,
                "dropout": dropout,
                "lr": learning_rate,
                "epochs": epochs,
            }
        )
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "elapsed_seconds": elapsed,
        }

        save_checkpoint(
            checkpoint_file,
            model_state_dict=model.state_dict(),
            tokenizer=tokenizer,
            config=config,
            metrics=metrics,
            seed=seed,
            extra={
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
        )
        torch.save(tokenizer, tokenizer_file)

    print(f"Modelo causal guardado en: {checkpoint_file}")
    print(f"Tokenizer guardado en: {tokenizer_file}")

    return model, tokenizer


if __name__ == "__main__":
    train_model()
