"""Metricas y utilidades de evaluacion para la practica."""

import math
from pathlib import Path

import torch

from fdi_pln_2608_p5.checkpoint import load_checkpoint, normalize_config
from fdi_pln_2608_p5.utils import resolve_device


def perplexity(loss):
    """Convierte cross entropy media en perplejidad."""
    return math.exp(loss)


def _flatten_valid(pred_ids, gold_ids, ignore_index=-100):
    pairs = []

    for pred_seq, gold_seq in zip(pred_ids, gold_ids):
        for pred, gold in zip(pred_seq, gold_seq):
            if int(gold) == ignore_index:
                continue
            pairs.append((int(pred), int(gold)))

    return pairs


def _safe_divide(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def compute_token_metrics(pred_ids, gold_ids, ignore_index=-100):
    """Calcula accuracy, precision, recall y F1 token-level sin sklearn.

    La precision/recall/F1 se calculan como micro-promedio sobre etiquetas de
    entidad, excluyendo la clase O. El padding se ignora con `ignore_index`.
    """

    pairs = _flatten_valid(pred_ids, gold_ids, ignore_index=ignore_index)
    if not pairs:
        return {
            "token_accuracy": None,
            "token_precision": None,
            "token_recall": None,
            "token_f1": None,
        }

    correct = sum(pred == gold for pred, gold in pairs)
    total = len(pairs)

    true_positive = sum(pred == gold and gold != 0 for pred, gold in pairs)
    false_positive = sum(pred != gold and pred != 0 for pred, gold in pairs)
    false_negative = sum(pred != gold and gold != 0 for pred, gold in pairs)

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )

    return {
        "token_accuracy": correct / total,
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }


def bio_tags_to_entities(tags):
    """Convierte una secuencia BIO en entidades exactas.

    Devuelve tuplas `(tipo, inicio, fin)`, con `fin` exclusivo. Si aparece un
    `I-X` sin entidad abierta compatible, se interpreta como inicio de entidad
    para hacer la metrica robusta ante salidas imperfectas del modelo.
    """

    entities = []
    current_type = None
    start = None

    for index, tag in enumerate(tags):
        if tag == "O":
            if current_type is not None:
                entities.append((current_type, start, index))
                current_type = None
                start = None
            continue

        prefix, entity_type = tag.split("-", maxsplit=1)

        if prefix == "B" or current_type != entity_type:
            if current_type is not None:
                entities.append((current_type, start, index))
            current_type = entity_type
            start = index

    if current_type is not None:
        entities.append((current_type, start, len(tags)))

    return entities


def compute_entity_metrics(pred_tags, gold_tags):
    """Calcula precision, recall y F1 por entidad exacta BIO."""

    pred_total = 0
    gold_total = 0
    true_positive = 0

    for pred_seq, gold_seq in zip(pred_tags, gold_tags):
        pred_entities = set(bio_tags_to_entities(pred_seq))
        gold_entities = set(bio_tags_to_entities(gold_seq))

        pred_total += len(pred_entities)
        gold_total += len(gold_entities)
        true_positive += len(pred_entities & gold_entities)

    precision = _safe_divide(true_positive, pred_total)
    recall = _safe_divide(true_positive, gold_total)
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )

    return {
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
    }


@torch.no_grad()
def evaluate_ner_dataloader(model, dataloader, device, ignore_index=-100):
    """Ejecuta inferencia NER sobre un dataloader y calcula metricas."""

    from fdi_pln_2608_p5.modules.ner import ID2LABEL

    model.eval()
    pred_ids = []
    gold_ids = []
    pred_tags = []
    gold_tags = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)
        batch_pred = logits.argmax(dim=-1).cpu().tolist()
        batch_gold = y.cpu().tolist()

        pred_ids.extend(batch_pred)
        gold_ids.extend(batch_gold)

        for pred_seq, gold_seq in zip(batch_pred, batch_gold):
            valid_pred_tags = []
            valid_gold_tags = []
            for pred, gold in zip(pred_seq, gold_seq):
                if gold == ignore_index:
                    continue
                valid_pred_tags.append(ID2LABEL[int(pred)])
                valid_gold_tags.append(ID2LABEL[int(gold)])
            pred_tags.append(valid_pred_tags)
            gold_tags.append(valid_gold_tags)

    metrics = compute_token_metrics(pred_ids, gold_ids, ignore_index=ignore_index)
    metrics.update(compute_entity_metrics(pred_tags, gold_tags))
    return metrics


def load_ner_checkpoint_model(weights, device=None):
    """Carga un checkpoint NER autocontenido."""

    from fdi_pln_2608_p5.modules.ner import NERLLM, NUM_LABELS

    device = resolve_device(device)
    checkpoint = load_checkpoint(weights, map_location=device)
    config = normalize_config(checkpoint["config"])
    tokenizer = checkpoint["tokenizer"]

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

    return model, tokenizer, config, device


def evaluate_ner_checkpoint(weights, data_path, batch_size=16, device=None):
    """Evalua un checkpoint NER contra un fichero CoNLL/BIO."""

    from torch.utils.data import DataLoader

    from fdi_pln_2608_p5.modules.ner import NERDataset, collate_ner
    from fdi_pln_2608_p5.modules.train_ner import load_ner_data

    model, tokenizer, config, device = load_ner_checkpoint_model(weights, device=device)
    ner_data = load_ner_data(data_path)
    dataset = NERDataset(
        ner_data=ner_data,
        tokenizer=tokenizer,
        max_len=config["context_size"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ner,
    )

    return evaluate_ner_dataloader(model, dataloader, device)


def analyze_bpe(weights, text=None, file_path=None):
    """Devuelve informacion de segmentacion BPE para texto o fichero."""

    if file_path is not None:
        text = Path(file_path).read_text(encoding="utf-8")
    if text is None:
        raise ValueError("Indica --text o --file para analizar BPE.")

    checkpoint = load_checkpoint(weights, map_location="cpu")
    tokenizer = checkpoint["tokenizer"]
    token_ids = tokenizer.encode(text)
    pieces = [tokenizer.decode([token_id]) for token_id in token_ids]
    n_chars = len(text)
    n_tokens = len(token_ids)

    return {
        "text": text,
        "token_ids": token_ids,
        "pieces": pieces,
        "n_chars": n_chars,
        "n_tokens": n_tokens,
        "chars_per_token": None if n_tokens == 0 else n_chars / n_tokens,
        "segmentation": " | ".join(piece.replace("\n", "\\n") for piece in pieces),
    }


__all__ = [
    "analyze_bpe",
    "bio_tags_to_entities",
    "compute_entity_metrics",
    "compute_token_metrics",
    "evaluate_ner_checkpoint",
    "evaluate_ner_dataloader",
    "load_ner_checkpoint_model",
    "perplexity",
]
