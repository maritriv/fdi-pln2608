# Modelo transformer mínimo para modelado del lenguaje

import torch
import torch.nn as nn
import torch.nn.functional as F

from fdi_pln_2608_p5.modules.attention import Attention


class FeedForward(nn.Module):
    """Perceptrón multicapa posición a posición."""

    def __init__(self, d_model, dropout, expansion=4):
        super().__init__()

        hidden_dim = expansion * d_model

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Bloque transformer pre-norm."""

    def __init__(self, d_model, n_heads, max_seq_len, dropout, expansion=4):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(
            d_model=d_model,
            dropout=dropout,
            expansion=expansion,
        )

    def forward(self, x, causal=True):
        x = x + self.attn(self.ln1(x), causal=causal)
        x = x + self.ff(self.ln2(x))
        return x


class MiniLLM(nn.Module):
    """Transformer decoder pequeño para modelado del lenguaje causal."""

    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        n_layers,
        max_seq_len,
        dropout,
        expansion=4,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model debe ser divisible por n_heads")

        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.expansion = expansion

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model=d_model,
                    n_heads=n_heads,
                    max_seq_len=max_seq_len,
                    dropout=dropout,
                    expansion=expansion,
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)

        # Head de lenguaje + weight tying con embeddings de entrada
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, x, targets=None, causal=True):
        """Devuelve (logits, loss).

        x: Tensor (batch_size, n_tokens)
        targets: Tensor (batch_size, n_tokens), opcional
        """
        batch_size, n_tokens = x.shape

        if n_tokens > self.max_seq_len:
            raise ValueError(
                f"La secuencia tiene longitud {n_tokens}, "
                f"pero max_seq_len={self.max_seq_len}"
            )

        positions = torch.arange(n_tokens, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, n_tokens)

        x = self.token_emb(x) + self.pos_emb(positions)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, causal=causal)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.flatten(0, 1),
            targets.flatten(),
        )

        return logits, loss

    @torch.no_grad()
    def generate(self, prompt, max_tokens=200, temperature=0.8, top_k=None):
        """Genera tokens a partir de un prompt.

        prompt: lista de ids de tokens.
        max_tokens: número máximo de tokens a generar.
        temperature: controla aleatoriedad.
        top_k: limita el muestreo a los k tokens más probables.
        """
        if temperature <= 0:
            raise ValueError("temperature debe ser mayor que 0")

        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k debe ser positivo")
            top_k = min(top_k, self.vocab_size)

        self.eval()

        ventana = torch.tensor(
            [prompt[-self.max_seq_len :]],
            dtype=torch.long,
            device=next(self.parameters()).device,
        )

        generados = []

        for _ in range(max_tokens):
            logits, _ = self(ventana, causal=True)
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(next_token_logits, top_k)
                min_value = values[:, -1].unsqueeze(-1)

                next_token_logits = torch.where(
                    next_token_logits < min_value,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            generados.append(next_token_id.item())

            ventana = torch.cat([ventana, next_token_id], dim=1)
            ventana = ventana[:, -self.max_seq_len :]

        return generados


if __name__ == "__main__":
    batch_size = 2
    n_tokens = 8
    vocab_size = 100
    d_model = 32
    n_heads = 4
    n_layers = 2
    max_seq_len = 16
    dropout = 0.1

    x = torch.randint(0, vocab_size, (batch_size, n_tokens))
    y = torch.randint(0, vocab_size, (batch_size, n_tokens))

    model = MiniLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )

    logits, loss = model(x, targets=y, causal=True)
    generated = model.generate(prompt=[1, 2, 3], max_tokens=10, top_k=5)

    print("Entrada:", x.shape)
    print("Logits:", logits.shape)
    print("Loss:", loss.item())
    print("Generados:", generated)
