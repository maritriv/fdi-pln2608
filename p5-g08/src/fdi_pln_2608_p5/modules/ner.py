"""Modelo NER, dataset y alineamiento palabra -> BPE."""

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.utils.data import Dataset

from fdi_pln_2608_p5.modules.model import MiniLLM


LABEL2ID = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-LOC": 3,
    "I-LOC": 4,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


def align_to_bpe(words, word_labels, tokenizer):
    """Alinea etiquetas BIO de palabras a sub-tokens BPE."""

    token_ids = []
    token_labels = []

    space_ids = tokenizer.encode(" ")

    for i, (word, label) in enumerate(zip(words, word_labels)):
        if i > 0:
            token_ids.extend(space_ids)
            token_labels.extend(["O"] * len(space_ids))

        word_ids = tokenizer.encode(word)
        token_ids.extend(word_ids)

        if label.startswith("B-"):
            inside_label = "I-" + label[2:]
            token_labels.append(label)
            token_labels.extend([inside_label] * (len(word_ids) - 1))
        else:
            token_labels.extend([label] * len(word_ids))

    return token_ids, token_labels


def encode_words_with_spans(words, tokenizer):
    """Codifica palabras y guarda que subtokens pertenecen a cada palabra."""

    token_ids = []
    word_spans = []
    space_ids = tokenizer.encode(" ")

    for i, word in enumerate(words):
        if i > 0:
            token_ids.extend(space_ids)

        start = len(token_ids)
        word_ids = tokenizer.encode(word)
        token_ids.extend(word_ids)
        end = len(token_ids)

        word_spans.append((start, end))

    return token_ids, word_spans


def explain_alignment(words, word_labels, tokenizer):
    """Muestra cómo se alinean palabras y etiquetas BIO con los sub-tokens BPE."""

    print(f"Frase: {' '.join(words)}")

    for word, label in zip(words, word_labels):
        ids = tokenizer.encode(word)
        pieces = [tokenizer.decode([i]) for i in ids]

        if label.startswith("B-"):
            inside_label = "I-" + label[2:]
            labels = [label] + [inside_label] * (len(ids) - 1)
        else:
            labels = [label] * len(ids)

        pairs = "  ".join(f"{piece}/{lab}" for piece, lab in zip(pieces, labels))
        print(f"{word:<15} {label:<6} -> {pairs}")


class NERLLM(MiniLLM):
    """MiniLLM reutilizado para NER.

    Usa el mismo backbone Transformer del modelo causal, pero sustituye la cabeza
    de lenguaje por una cabeza de clasificación BIO por token.

    Para NER se usa atención bidireccional: causal=False.
    """

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model,
        n_heads,
        n_layers,
        dropout,
        num_labels=NUM_LABELS,
        expansion=4,
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
            expansion=expansion,
        )

        self.num_labels = num_labels
        self.ner_head = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, labels=None):
        """Devuelve logits y loss opcional.

        input_ids: Tensor (batch_size, n_tokens)
        labels: Tensor (batch_size, n_tokens), con -100 en padding
        """

        batch_size, n_tokens = input_ids.shape

        if n_tokens > self.max_seq_len:
            raise ValueError(
                f"La secuencia tiene longitud {n_tokens}, "
                f"pero max_seq_len={self.max_seq_len}"
            )

        positions = torch.arange(n_tokens, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, n_tokens)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, causal=False)

        hidden = self.ln_f(x)
        logits = self.ner_head(hidden)

        loss = None

        if labels is not None:
            loss = cross_entropy(
                logits.flatten(0, 1),
                labels.flatten(),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def predict_entities(self, words, tokenizer):
        """Predice entidades nombradas sobre una lista de palabras.

        Devuelve una lista de pares:
        [(texto_entidad, tipo), ...]
        """

        self.eval()

        ids, word_spans = encode_words_with_spans(words, tokenizer)

        device = next(self.parameters()).device
        offset = max(0, len(ids) - self.max_seq_len)

        input_ids = torch.tensor(
            [ids[offset:]],
            dtype=torch.long,
            device=device,
        )

        logits, _ = self(input_ids)
        pred_ids = logits.argmax(dim=-1)[0].tolist()
        pred_labels = [ID2LABEL[pred_id] for pred_id in pred_ids]

        entities = []
        current_words = []
        current_type = None

        for word, (start, end) in zip(words, word_spans):
            if end <= offset:
                continue

            start = max(start - offset, 0)
            end = min(end - offset, len(pred_labels))
            if start >= end:
                continue

            label = self._word_label(pred_labels[start:end])

            if label == "O":
                if current_words:
                    entities.append((" ".join(current_words), current_type))
                    current_words = []
                    current_type = None
                continue

            entity_type = label[2:]

            if label.startswith("B-") or current_type != entity_type:
                if current_words:
                    entities.append((" ".join(current_words), current_type))
                current_words = [word]
                entity_type = label[2:]
                current_type = entity_type
            else:
                current_words.append(word)

        if current_words:
            entities.append((" ".join(current_words), current_type))

        return entities

    @staticmethod
    def _word_label(subtoken_labels):
        """Resume etiquetas de subtokens a una etiqueta BIO de palabra."""
        counts = {}
        for label in subtoken_labels:
            if label == "O":
                continue
            entity_type = label[2:]
            counts[entity_type] = counts.get(entity_type, 0) + 1

        if counts:
            entity_type = max(counts, key=counts.get)
            return f"B-{entity_type}"

        return "O"


class NERDataset(Dataset):
    """Dataset de NER con etiquetas BIO alineadas a BPE."""

    def __init__(self, ner_data, tokenizer, max_len=128):
        self.samples = []

        for words, labels in ner_data:
            ids, token_labels = align_to_bpe(words, labels, tokenizer)

            ids = ids[:max_len]
            token_labels = token_labels[:max_len]

            self.samples.append(
                (
                    torch.tensor(ids, dtype=torch.long),
                    torch.tensor(
                        [LABEL2ID[label] for label in token_labels],
                        dtype=torch.long,
                    ),
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_ner(batch):
    """Hace padding de un batch NER.

    Las etiquetas de padding son -100 para que cross_entropy las ignore.
    """

    xs, ys = zip(*batch)

    max_len = max(len(x) for x in xs)

    padded_x = torch.zeros(len(xs), max_len, dtype=torch.long)
    padded_y = torch.full((len(ys), max_len), -100, dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_x[i, : len(x)] = x
        padded_y[i, : len(y)] = y

    return padded_x, padded_y


def extract_named_entities(text):
    """Baseline simple de NER por mayúsculas.

    Se conserva solo como alternativa auxiliar. La entrega final usa NERLLM.
    """
    import re

    stopwords = {
        "The",
        "A",
        "An",
        "And",
        "But",
        "Or",
        "If",
        "Then",
        "When",
        "While",
        "It",
        "He",
        "She",
        "They",
        "We",
        "I",
        "You",
        "This",
        "That",
        "In",
        "On",
        "At",
        "For",
        "Of",
        "To",
        "From",
    }

    pattern = r"\b(?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+)*\b"
    candidates = re.findall(pattern, text)

    entities = []
    seen = set()

    for candidate in candidates:
        if candidate in stopwords:
            continue

        if candidate not in seen:
            entities.append(candidate)
            seen.add(candidate)

    return entities


if __name__ == "__main__":
    example_words = ["Alice", "met", "the", "Queen", "in", "Wonderland"]
    example_labels = ["B-PER", "O", "O", "B-PER", "O", "B-LOC"]

    print("Este módulo define el modelo NER.")
    print("Para probar explain_alignment necesitas cargar un tokenizer entrenado.")
