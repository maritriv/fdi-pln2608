# Práctica 5 - Mini LLM causal

## Objetivo
Implementar, entrenar y evaluar un pequeño LLM causal basado en Transformer.

--- 

## Estructura
- `attention.py`: atención multi-cabezal causal.
- `model.py`: Transformer + cabeza de lenguaje.
- `tokenizer.py`: tokenizador BPE.
- `data.py`: carga del corpus y datasets.
- `train.py`: entrenamiento y validación.
- `generate.py`: generación de texto.
- `ner.py`: extracción de entidades nombradas.
- `experiment.py`: experimentos de hiperparámetros.

---

## Entrenamiento
```bash
uv run fdi-pln-2608-p5 train --resources resources --epochs 5
```
---

## Generación
```bash
uv run fdi-pln-2608-p5 generate --prompt "Alice was" --max-new-tokens 100
```
---

## NER
```bash
uv run fdi-pln-2608-p5 ner --text "Alice met the Queen in Wonderland."
```

---