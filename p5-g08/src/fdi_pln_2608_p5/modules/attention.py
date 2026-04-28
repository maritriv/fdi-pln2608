# Mecanismo de atención comentado
#
# PLN 2025/2026 (FDI UCM)
# Antonio F. G. Sevilla <afgs@ucm.es>

import math

import torch
import torch.nn as nn
from torch.nn.functional import softmax


class Attention(nn.Module):
    """Auto-atención multi-cabezal con escala (scaled multi-head self-attention)

    Si `causal=True` en el forward, cada posicion solo atiende a las
    anteriores (util para generacion). Si `causal=False`, cada posicion
    atiende a toda la secuencia (TAREA: para qué querríamos esto?).

    dropout es el porcentaje de dropout a usar.
    """

    def __init__(self, d_model, n_heads, max_seq_len, dropout):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model debe ser divisible por n_heads")

        self.n_heads = n_heads

        # Distribuimos la dimensión del modelo entre el numero de cabezas
        self.head_dim = d_model // n_heads

        # Una única matriz para QKV, luego separaremos
        self.qkv = nn.Linear(d_model, 3 * d_model)

        # Capa lineal para permitir al modelo reproyectar los vectores contexto
        self.out = nn.Linear(d_model, d_model)

        # El dropout se activa en train y desactiva en test gracias a pytorch
        self.dropout = nn.Dropout(dropout)

        # La máscara causal pone a -inf las posiciones correspondientes a tokens
        # "futuros" (triangular superior)
        mask = torch.triu(
            torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1
        )

        # Registramos la máscara causal como tensor (no entrenable)
        self.register_buffer("mask", mask)

    def forward(self, x, causal=True):
        # x tiene forma:
        # (batch_size, n_tokens, d_model)
        _, n_tokens, _ = x.shape

        # Multiplicamos x por QKV (todo junto), pero separamos a lo largo de la
        # última dimensión para tener las matrices de queries, keys y values
        q, k, v = self.qkv(x).tensor_split(3, dim=-1)

        # Separamos en cabezales
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # q, k, v tienen forma:
        # (batch_size, n_heads, n_tokens, head_dim)

        # 1. Calculamos la atención:
        # producto matricial entre queries y keys transpuestas
        # resultado: (batch_size, n_heads, n_tokens, n_tokens)
        att = q @ k.transpose(-2, -1)

        # 2. Escalamos por la raíz de la dimensión de cada cabeza
        # para que los logits no crezcan demasiado
        att = att / math.sqrt(self.head_dim)

        # 3. Si la atención es causal, impedimos mirar al futuro
        if causal:
            att = att + self.mask[:n_tokens, :n_tokens]

        # 4. Normalizamos la atención para que sume 1
        att = softmax(att, dim=-1)

        # 5. Aplicamos dropout sobre la matriz de atención
        att = self.dropout(att)

        # 6. Usamos la atención para combinar los values
        # resultado: (batch_size, n_heads, n_tokens, head_dim)
        z = att @ v

        # "Deshacemos" la partición en cabezales
        # (batch_size, n_heads, n_tokens, head_dim) -> (batch_size, n_tokens, d_model)
        z = z.transpose(1, 2).flatten(-2)

        # Re-proyectamos con la última transformación
        return self.out(z)

    def split_heads(self, x):
        # (batch_size, n_tokens, d_model) -> (batch_size, n_tokens, n_heads, head_dim)
        # "partimos" la última dimensión (-1, d_model) en n_heads x head_dim
        x = x.unflatten(-1, (self.n_heads, self.head_dim))
        # (batch_size, n_tokens, n_heads, head_dim) -> (batch_size, n_heads, n_tokens, head_dim)
        #  transponemos n_tokens y n_heads para que cada cabezal de atención se
        # "multiplique por separado", haciéndolos independientes
        # TAREA: hacer el álgebra a mano para ver como funciona
        return x.transpose(1, 2)
