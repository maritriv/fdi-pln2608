# Tokenizador BPE (Byte Pair Encoding) mínimo, entrenado sobre el texto."""
#
# PLN 2025/2026 (FDI UCM)
# Antonio F. G. Sevilla <afgs@ucm.es>


from collections import Counter


class BPETokenizer:
    """Byte Pair Encoding entrenado sobre un texto.

    Vocabulario inicial: caracteres unicos del texto. Durante el
    entrenamiento se buscan los pares adyacentes mas frecuentes y se
    fusionan en nuevos tokens, hasta alcanzar `vocab_size` tokens.

    NOTA: para ser BPE de verdad, tendríamos que hacerlo sobre bytes, no sobre
    caracteres, pero para la práctica funciona bien.
    """

    def __init__(self, text, vocab_size=300):
        self.vocab_size = vocab_size
        # 1. Inicializamos con caracteres encontrados en el texto
        self.vocab = sorted(set(text))  # vocab[id] -> token string.
        self.tok2id = {tok: i for i, tok in enumerate(self.vocab)}

        # 2. Convertimos el texto en ids de caracteres
        tokens = [self.tok2id[c] for c in text]
        # # Aquí guardamos las fusiones aprendidas
        self.merges = []  # lista de ((id_a, id_b), nuevo_id), para encode()

        # Aprendizaje BPE
        for new_id in range(len(self.vocab), vocab_size):
            # Contamos pares consecutivos (bigramas)
            pairs = Counter(zip(tokens, tokens[1:]))

            # Si no hay pares, terminamos
            if not pairs:
                break

            # Cogemos el par más frecuente
            best = pairs.most_common(1)[0][0]

            # Creamos nuevo token concatenando strings
            new_tok = self.vocab[best[0]] + self.vocab[best[1]]

            # Añadimos al vocabulario
            self.tok2id[new_tok] = new_id
            self.vocab.append(new_tok)
            self.merges.append((best, new_id))

            tokens = self._apply_merge(tokens, best[0], best[1], new_id)

    @staticmethod
    def _apply_merge(tokens, a, b, new_id):
        """Reemplaza todas las ocurrencias del par (a, b) por new_id."""
        merged = []
        i = 0

        while i < len(tokens):
            # Si encontramos el par (a, b), lo sustituimos
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                merged.append(new_id)
                i += 2  # saltamos ambos
            else:
                merged.append(tokens[i])
                i += 1

        return merged

    def encode(self, text):
        """Codifica un texto aplicando los merges aprendidos."""

        # 1. Convertir a caracteres
        # Si un carácter no apareció en entrenamiento, lo sustituimos por el id 0.
        # En esta implementación mínima, 0 no es un token especial sino el primer
        # token del vocabulario.
        tokens = [self.tok2id.get(c, 0) for c in text]

        # Aplicar merges
        for (a, b), new_id in self.merges:
            tokens = self._apply_merge(tokens, a, b, new_id)
        return tokens

    def decode(self, ids):
        """Decodifica una lista de ids a texto."""
        return "".join(self.vocab[i] for i in ids)

    def __repr__(self):
        """Representación más bonita del vocabulario."""
        pretty = [t.replace("\n", "\\n").replace(" ", "▁") for t in self.vocab]
        return f"{len(self.vocab)} tokens: {pretty}"


# Si ejecutamos este módulo directamente, probamos el tokenizador
if __name__ == "__main__":
    import sys
    from pathlib import Path

    files_path = Path(sys.argv[1] if len(sys.argv) > 1 else "resources")
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    textos = "\n\n".join(
        open(p, encoding="utf-8").read() for p in files_path.glob("*.txt")
    )
    tokenizer = BPETokenizer(textos, vocab_size=vocab_size)
    print(tokenizer)
