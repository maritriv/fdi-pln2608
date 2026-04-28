from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, random_split

from fdi_pln_2608_p5.modules.ner import NERDataset, NERLLM, NUM_LABELS, collate_ner


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

    return total_loss / len(dataloader)


def train_ner_model(
    ner_data_path,
    causal_model_path="checkpoints/p5_causal_26XX.pth",
    tokenizer_path="checkpoints/tokenizer.pt",
    save_path="checkpoints/p5_ner_26XX.pth",
    batch_size=16,
    epochs=10,
    learning_rate=3e-4,
    train_ratio=0.9,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Usando dispositivo: {device}")

    tokenizer = torch.load(tokenizer_path, map_location=device)
    causal_checkpoint = torch.load(causal_model_path, map_location=device)
    config = causal_checkpoint["config"]

    ner_data = load_ner_data(ner_data_path)

    dataset = NERDataset(
        ner_data=ner_data,
        tokenizer=tokenizer,
        max_len=config["seq_len"],
    )

    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

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
        max_seq_len=config["seq_len"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        num_labels=NUM_LABELS,
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

    for epoch in range(1, epochs + 1):
        train_loss = run_ner_epoch(model, train_loader, device, optimizer)
        val_loss = run_ner_epoch(model, val_loader, device)

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{epochs} | "
            f"ner_train_loss={train_loss:.4f} | "
            f"ner_val_loss={val_loss:.4f} | "
            f"tiempo={elapsed:.1f}s"
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "num_labels": NUM_LABELS,
        },
        save_path,
    )

    print(f"Modelo NER guardado en: {save_path}")

    return model, tokenizer
