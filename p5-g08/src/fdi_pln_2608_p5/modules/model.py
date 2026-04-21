# Modelo transformer mínimo para modelado del lenguaje

import torch
import torch.nn as nn

from fdi_pln_2608_p5.modules.attention import Attention


class FeedForward(nn.Module):
    """Perceptrón multicapa posición a posición.

    Se aplica de forma independiente a cada token:
    - expande la dimensión del embedding
    - aplica no linealidad
    - vuelve a proyectar a d_model

    Esto añade capacidad de abstracción al bloque transformer.
    """

    def __init__(self, d_model, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Bloque transformer pre-norm.

    Estructura:
    1. LayerNorm
    2. Self-attention
    3. Conexión residual
    4. LayerNorm
    5. Feed-forward
    6. Conexión residual
    """

    def __init__(self, d_model, n_heads, max_seq_len, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, dropout=dropout)

    def forward(self, x, causal=True):
        # Primer sub-bloque: atención + residual
        x = x + self.attn(self.ln1(x), causal=causal)

        # Segundo sub-bloque: feed-forward + residual
        x = x + self.ff(self.ln2(x))

        return x


class MiniLLM(nn.Module):
    """Transformer decoder pequeño para modelado del lenguaje.

    Entrada:
    - ids de tokens con forma (batch_size, n_tokens)

    Salida:
    - logits con forma (batch_size, n_tokens, vocab_size)

    Cada posición produce una distribución sobre el siguiente token.
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        n_layers,
        max_seq_len,
        dropout,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embeddings de tokens: convierten ids en vectores densos
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Embeddings posicionales aprendidos
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Pila de bloques transformer
        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model=d_model,
                    n_heads=n_heads,
                    max_seq_len=max_seq_len,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # Normalización final antes de proyectar al vocabulario
        self.ln_f = nn.LayerNorm(d_model)

        # Cabeza de lenguaje: proyecta cada vector al tamaño del vocabulario
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Añadir weight tying: compartir pesos entre token_emb y lm_head (mejora la generalización y reduce el número de parámetros)
        self.lm_head.weight = self.token_emb.weight # Nota: Esto solo funciona si d_model es igual al tamaño del embedding, que en nuestro caso lo es

    def forward(self, x, causal=True):
        # x tiene forma: (batch_size, n_tokens)
        batch_size, n_tokens = x.shape

        if n_tokens > self.max_seq_len:
            raise ValueError(
                f"La secuencia tiene longitud {n_tokens}, "
                f"pero max_seq_len={self.max_seq_len}"
            )

        # Creamos índices de posición: [0, 1, 2, ..., n_tokens-1]
        positions = torch.arange(n_tokens, device=x.device)

        # Expandimos a todo el batch:
        # (n_tokens,) -> (batch_size, n_tokens)
        positions = positions.unsqueeze(0).expand(batch_size, n_tokens)

        # Sumamos embeddings de token + embeddings posicionales
        x = self.token_emb(x) + self.pos_emb(positions)

        # Aplicamos los bloques transformer
        for block in self.blocks:
            x = block(x, causal=causal)

        # Normalización final
        x = self.ln_f(x)

        # Proyección final a logits sobre el vocabulario
        logits = self.lm_head(x)

        return logits


if __name__ == "__main__":
    # Prueba rápida del modelo

    batch_size = 2
    n_tokens = 8
    vocab_size = 100
    d_model = 32
    n_heads = 4
    n_layers = 2
    max_seq_len = 16
    dropout = 0.1

    # Batch de ids aleatorios
    x = torch.randint(0, vocab_size, (batch_size, n_tokens))

    model = MiniLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )

    logits = model(x, causal=True)

    print("Entrada:", x.shape)
    print("Logits:", logits.shape)
