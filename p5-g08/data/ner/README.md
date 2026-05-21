# Datos NER

Esta carpeta contiene el corpus NER final integrado desde la preentrega:

- `merged.json`: copia del corpus fusionado de `../pre-entrega_2601/merged.json`.
- `final.conll`: conversion reproducible a formato CoNLL/BIO.

Para regenerar el fichero CoNLL:

```bash
uv run fdi-pln-2608-p5 prepare-ner-data \
  --input ../pre-entrega_2601/merged.json \
  --output data/ner/final.conll
```

El mapeo de etiquetas es:

| Preentrega | BIO |
| --- | --- |
| `o` | `O` |
| `pi` | `B-PER` |
| `pc` | `I-PER` |
| `li` | `B-LOC` |
| `lc` | `I-LOC` |

Ejemplo de formato:

```text
Alice B-PER
met O
the O
Queen B-PER

Wonderland B-LOC
```

Nombre recomendado para la entrega:

```text
data/ner/final.conll
```
