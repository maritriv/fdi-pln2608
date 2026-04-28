# Práctica 5 - Mini LLM causal + NER

## Objetivo

Implementar desde cero un pequeño modelo de lenguaje (LLM) basado en Transformer, entrenarlo para generación de texto y reutilizar su arquitectura para resolver una tarea de reconocimiento de entidades nombradas (NER).

---

## Funcionalidades

El proyecto incluye:

- Implementación completa de un Transformer decoder:
  - Atención multi-cabezal causal
  - Feed-forward
  - LayerNorm (pre-norm)
- Tokenizador BPE entrenado desde texto
- Entrenamiento de modelo causal (language modeling)
- Generación de texto
- Modelo NER basado en el mismo backbone
- Entrenamiento específico para NER
- Predicción de entidades desde texto o fichero

---

## Estructura del proyecto
```bash
fdi_pln_2608_p5/
│
├── modules/
│ ├── attention.py # Multi-head attention
│ ├── model.py # MiniLLM (Transformer)
│ ├── tokenizer.py # BPE tokenizer
│ ├── data.py # Dataset LM
│ ├── train.py # Entrenamiento modelo causal
│ ├── generate.py # Generación de texto
│ ├── ner.py # Modelo NER + dataset
│ ├── train_ner.py # Entrenamiento NER
│ ├── ner_predict.py # Inferencia NER
│ └── experiments.py # Experimentos (opcional)
│
├── resources/ # Corpus y datos
├── checkpoints/ # Modelos entrenados
├── main.py # CLI principal
└── README.md
```


---

## Instalación y ejecución

El proyecto está preparado para ejecutarse con `uv`.

También esta formateado con:

```bash
uv format
uv format --check
```

## Entrenamiento del modelo causal

Entrena el modelo de lenguaje:

```bash
uv run fdi-pln-2608-p5 train --resources resources --epochs 5
```

Salida:

- `checkpoints/p5_causal_26XX.pth`
- `checkpoints/tokenizer.pt`

## Generación de texto

```bash
uv run fdi-pln-2608-p5 generate --prompt "Alice was" --max-new-tokens 100  --top-k 20
```

Parámetros importantes:

- `temperature`: controla aleatoriedad
- `top-k`: limita vocabulario

## Entrenamiento del modelo NER

Se entrena reutilizando el modelo causal:
```bash
uv run fdi-pln-2608-p5 train-ner --ner-data resources/ner_train.tsv --epochs 10
```

Salida:

- `checkpoints/p5_ner_26XX.pth`

## Detección de entidades nombradas
**Desde texto**

```bash
uv run fdi-pln-2608-p5 ner --text "Alice met the Queen in Wonderland"
```

**Desde fichero:**
```bash
uv run fdi-pln-2608-p5 ner --file ejemplo.txt
```

## Formato de datos NER

Formato esperado (tipo CoNLL):
```bash
Alice B-PER
met O
the O
Queen B-PER

Wonderland B-LOCç

```

## Decisiones de diseño

- Se usa arquitectura Transformer decoder (modelo causal)
- Weight tying entre embeddings y salida
- Atención causal para generación
- Atención bidireccional para NER
- Tokenización BPE aprendida del corpus
- Reutilización del backbone para NER

---

## Experimentos (opcional)

Se han probado diferentes configuraciones:

- Modelos más profundos
- Distintos tamaños de embedding
- Distinto dropout

---

## Resultados

- El modelo es capaz de generar texto coherente basado en el corpus
- El modelo NER detecta entidades con arquitectura compartida
- Se observa mejora al aumentar capas o entrenamiento

---

## Archivos entregados

- `fdi_pln_26XX_p5-1.0-py3-none-any.whl`
- `p5_causal_26XX.pth`
- `p5_ner_26XX.pth`
- `README.md`

---

## Autores

- Marina Triviño de las Heras
- Carlota Salazar Martín

