from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, random_split

from fdi_pln_2608_p5.checkpoint import (
    load_checkpoint,
    normalize_config,
    save_checkpoint,
)
from fdi_pln_2608_p5.evaluate import evaluate_ner_dataloader
from fdi_pln_2608_p5.modules.ner import NERDataset, NERLLM, NUM_LABELS, collate_ner
from fdi_pln_2608_p5.utils import resolve_device, set_seed


def load_ner_data(path):
    """Carga datos NER en formato CoNLL.

    Formato esperado:
    Alice B-PER
    met O
    the O
    Queen B-PER

    Wonderland B-LOC
    ...
    """
    data = []
    words = []
    labels = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if words:
                    data.append((words, labels))
                    words = []
                    labels = []
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            word = " ".join(parts[:-1])
            label = parts[-1]

            words.append(word)
            labels.append(label)

    if words:
        data.append((words, labels))

    if not data:
        raise ValueError(f"No se encontraron datos NER válidos en {path}")

    return data


def run_ner_epoch(model, dataloader, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            if is_train:
                optimizer.zero_grad()

            _, loss = model(x, labels=y)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


def train_ner_model(
    ner_data_path,
    causal_model_path="checkpoints/p5_causal_2608.pth",
    tokenizer_path=None,
    save_path="checkpoints/p5_ner_2608.pth",
    batch_size=16,
    epochs=10,
    learning_rate=3e-4,
    train_ratio=0.9,
    seed=42,
    device=None,
):
    set_seed(seed)
    device = resolve_device(device)

    print(f"Usando dispositivo: {device}")

    causal_checkpoint = load_checkpoint(causal_model_path, map_location=device)
    tokenizer = causal_checkpoint.get("tokenizer")
    if tokenizer is None:
        if tokenizer_path is None:
            raise ValueError(
                "El checkpoint causal no contiene tokenizer; indica --tokenizer-path."
            )
        tokenizer = load_checkpoint(tokenizer_path, map_location=device)
    config = normalize_config(causal_checkpoint["config"])

    ner_data = load_ner_data(ner_data_path)

    dataset = NERDataset(
        ner_data=ner_data,
        tokenizer=tokenizer,
        max_len=config["context_size"],
    )

    train_size = max(1, int(len(dataset) * train_ratio))
    val_size = len(dataset) - train_size
    if val_size == 0:
        train_size = len(dataset)

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_ner,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ner,
    )

    model = NERLLM(
        vocab_size=config["vocab_size"],
        max_seq_len=config["context_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        num_labels=NUM_LABELS,
        expansion=config.get("expansion", 4),
    ).to(device)

    missing, unexpected = model.load_state_dict(
        causal_checkpoint["model_state_dict"],
        strict=False,
    )

    print("Backbone cargado desde modelo causal.")
    print(f"Pesos no cargados esperados: {missing}")
    print(f"Pesos inesperados ignorados: {unexpected}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    t0 = time.time()
    metrics = {}

    for epoch in range(1, epochs + 1):
        train_loss = run_ner_epoch(model, train_loader, device, optimizer)
        val_loss = (
            run_ner_epoch(model, val_loader, device) if val_size else float("nan")
        )

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{epochs} | "
            f"ner_train_loss={train_loss:.4f} | "
            f"ner_val_loss={val_loss:.4f} | "
            f"tiempo={elapsed:.1f}s"
        )
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "elapsed_seconds": elapsed,
            "token_accuracy": None,
            "token_precision": None,
            "token_recall": None,
            "token_f1": None,
            "entity_precision": None,
            "entity_recall": None,
            "entity_f1": None,
        }

        if val_size:
            metrics.update(evaluate_ner_dataloader(model, val_loader, device))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ner_config = normalize_config(
        {
            **config,
            "task": "ner_bio",
            "num_labels": NUM_LABELS,
            "batch_size": batch_size,
            "lr": learning_rate,
            "epochs": epochs,
        }
    )

    save_checkpoint(
        save_path,
        model_state_dict=model.state_dict(),
        tokenizer=tokenizer,
        config=ner_config,
        metrics=metrics,
        seed=seed,
        extra={
            "num_labels": NUM_LABELS,
            "causal_checkpoint": str(causal_model_path),
        },
    )

    print(f"Modelo NER guardado en: {save_path}")

    return model, tokenizer
