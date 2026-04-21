# Utilidades de datos para entrenar el mini LLM

from pathlib import Path

import torch
from torch.utils.data import Dataset

from fdi_pln_2608_p5.modules.tokenizer import BPETokenizer


def load_corpus(resources_path):
    """Lee todos los .txt de una carpeta y los concatena en un solo string."""
    files_path = Path(resources_path)
    texts = []

    for path in sorted(files_path.glob("*.txt")):
        texts.append(path.read_text(encoding="utf-8"))

    return "\n\n".join(texts)


class LanguageModelingDataset(Dataset):
    """Dataset para modelado del lenguaje autoregresivo.

    A partir de una secuencia larga de ids, crea ejemplos:
    - x: ventana de longitud seq_len
    - y: la misma ventana desplazada una posición a la derecha

    Ejemplo:
    ids = [10, 11, 12, 13, 14]
    seq_len = 3

    x = [10, 11, 12]
    y = [11, 12, 13]
    """

    def __init__(self, token_ids, seq_len):
        # Convertimos a tensor aquí para que el entrenamiento sea más rápido
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        # Necesitamos al menos seq_len + 1 tokens para construir (x, y)
        return max(0, len(self.token_ids) - self.seq_len)

    def __getitem__(self, idx):
        x = self.token_ids[idx : idx + self.seq_len]
        y = self.token_ids[idx + 1 : idx + self.seq_len + 1]

        return x, y


def build_tokenizer_and_dataset(resources_path, vocab_size, seq_len, train_ratio=0.9):
    """Carga corpus, entrena tokenizer y construye dataset de entrenamiento."""
    text = load_corpus(resources_path)

    tokenizer = BPETokenizer(text, vocab_size=vocab_size)
    token_ids = tokenizer.encode(text)

    # --- CAMBIO AQUÍ: División de datos ---
    split_idx = int(len(token_ids) * train_ratio)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    train_dataset = LanguageModelingDataset(train_ids, seq_len=seq_len)
    val_dataset = LanguageModelingDataset(val_ids, seq_len=seq_len)
    
    # Devolvemos ambos datasets
    return tokenizer, train_dataset, val_dataset, text


if __name__ == "__main__":
    resources_path = "resources"
    vocab_size = 200
    seq_len = 32

    tokenizer, dataset, text, token_ids = build_tokenizer_and_dataset(
        resources_path=resources_path,
        vocab_size=vocab_size,
        seq_len=seq_len,
    )

    print("Longitud del corpus (caracteres):", len(text))
    print("Tamaño del vocabulario:", len(tokenizer.vocab))
    print("Número total de tokens:", len(token_ids))
    print("Número de ejemplos en dataset:", len(dataset))

    x, y = dataset[0]
    print("Forma de x:", x.shape)
    print("Forma de y:", y.shape)
    print("Primer x:", x.tolist())
    print("Primer y:", y.tolist())
