# P5: Toves, Borogoves & Mome Raths

Practica 5 de Procesamiento del Lenguaje Natural. El proyecto implementa desde
cero un mini LLM basado en Transformers para generacion causal de texto y
reutiliza su backbone preentrenado para NER con etiquetado BIO.

## Integrantes

- Marina Trivino de las Heras
- Carlota Salazar Martin

## Arquitectura

El repositorio separa las piezas principales del sistema:

- `tokenizer.py`: tokenizador BPE entrenado sobre el corpus.
- `attention.py`: scaled multi-head self-attention con mascara causal opcional.
- `transformer.py`: embeddings de token y posicion, bloques Transformer pre-norm,
  residuales, LayerNorm, GELU y dropout.
- `causal_llm.py`: modelo causal con `lm_head` y weight tying.
- `ner_llm.py`: modelo BIO que reutiliza el Transformer y cambia la cabeza final.
- `checkpoint.py`: checkpoints autocontenidos con pesos, tokenizador, config,
  metricas y semilla.
- `cli.py`: comandos reproducibles de entrenamiento e inferencia.

La implementacion interna se mantiene en `src/fdi_pln_2608_p5/modules/` y la
raiz del paquete ofrece fachadas con nombres estables para una lectura mas clara.

## Instalacion

```bash
uv sync
```

El proyecto usa solo dependencias permitidas en la practica: `torch`, `numpy`,
`click`, `typer`, `loguru`, `rich`, `dynaconf` y `python-dotenv`.

## Comandos principales

```bash
uv run fdi-pln-2608-p5 --help
uv run fdi-pln-2608-p5 train-causal --help
uv run fdi-pln-2608-p5 train-ner --help
uv run fdi-pln-2608-p5 generate --help
uv run fdi-pln-2608-p5 ner --help
uv run fdi-pln-2608-p5 eval-ner --help
uv run fdi-pln-2608-p5 analyze-bpe --help
uv run fdi-pln-2608-p5 experiment-generate --help
```

## Entrenamiento causal

Configuracion CPU-friendly por defecto:

- `vocab_size=400`
- `context_size=128`
- `d_model=128`
- `n_heads=4`
- `n_layers=4`
- `expansion=4`
- `dropout=0.1`
- `batch_size=32`
- `lr=3e-4`
- `epochs=8`
- `seed=42`

```bash
uv run fdi-pln-2608-p5 train-causal \
  --corpus resources \
  --output checkpoints/p5_causal_2608.pth
```

Para una prueba mas rapida en CPU:

```bash
uv run fdi-pln-2608-p5 train-causal \
  --corpus resources \
  --output checkpoints/p5_causal_2608.pth \
  --context-size 64 \
  --n-layers 3 \
  --epochs 5
```

## Generacion

El checkpoint causal contiene tokenizador, configuracion, pesos, metricas y
semilla. Por eso la inferencia solo necesita `--weights`:

```bash
uv run fdi-pln-2608-p5 generate \
  --weights checkpoints/p5_causal_2608.pth \
  --prompt "Alice was" \
  --max-new-tokens 80 \
  --top-k 20
```

`temperature` regula la aleatoriedad del muestreo. Valores bajos como `0.5`
hacen la salida mas conservadora; valores altos como `1.2` exploran mas.

## Entrenamiento NER

El modelo NER carga el checkpoint causal, reutiliza embeddings y bloques
Transformer, y sustituye la cabeza de lenguaje por una cabeza BIO por token.

```bash
uv run fdi-pln-2608-p5 train-ner \
  --data data/ner/final.conll \
  --causal-weights checkpoints/p5_causal_2608.pth \
  --output checkpoints/p5_ner_2608.pth
```

Formato CoNLL esperado:

```text
Alice B-PER
met O
the O
Queen B-PER
in O
Wonderland B-LOC
```

Las etiquetas implementadas son `O`, `B-PER`, `I-PER`, `B-LOC` e `I-LOC`.

## Extraccion NER

```bash
uv run fdi-pln-2608-p5 ner \
  --weights checkpoints/p5_ner_2608.pth \
  --file examples/text.txt
```

Tambien se puede pasar texto directo:

```bash
uv run fdi-pln-2608-p5 ner \
  --weights checkpoints/p5_ner_2608.pth \
  --text "Alice met the Queen in Wonderland"
```

## Evaluacion NER

El comando `eval-ner` carga un checkpoint NER autocontenido, lee un fichero
CoNLL/BIO y calcula metricas sin dependencias externas:

- token accuracy
- token precision, recall y F1, excluyendo la clase `O`
- entity precision, recall y F1 con coincidencia exacta BIO

```bash
uv run fdi-pln-2608-p5 eval-ner \
  --weights checkpoints/p5_ner_2608.pth \
  --data data/ner/final.conll
```

Durante `train-ner`, si existe particion de validacion, estas metricas tambien
se guardan dentro del checkpoint en el campo `metrics`.

## Analisis BPE

Para justificar la tokenizacion en el informe:

```bash
uv run fdi-pln-2608-p5 analyze-bpe \
  --weights checkpoints/p5_causal_2608.pth \
  --text "Alice went to Wonderland"
```

Tambien acepta ficheros:

```bash
uv run fdi-pln-2608-p5 analyze-bpe \
  --weights checkpoints/p5_causal_2608.pth \
  --file examples/text.txt
```

El comando muestra ids, piezas decodificadas, numero de caracteres, numero de
tokens, ratio caracteres/tokens y una segmentacion legible.

## Experimentos de generacion

El comando `experiment-generate` no entrena nada. Ejecuta una rejilla pequena
con `temperature` en `0.5, 0.8, 1.2` y `top-k` en `10, 20, 50`, y guarda un
markdown para completar manualmente:

```bash
uv run fdi-pln-2608-p5 experiment-generate \
  --weights checkpoints/p5_causal_2608.pth \
  --prompt "Alice was" \
  --out reports/generation_experiments.md
```

## Checkpoints

Los checkpoints finales deben llamarse:

- `checkpoints/p5_causal_2608.pth`
- `checkpoints/p5_ner_2608.pth`

Formato:

```python
{
    "model_state_dict": ...,
    "tokenizer": ...,
    "config": ...,
    "metrics": ...,
    "seed": 42,
}
```

Esto permite reconstruir el modelo sin depender de un fichero externo de
tokenizador. Se conserva compatibilidad con checkpoints antiguos mediante
`--tokenizer-path` si hiciera falta.

## Decisiones tecnicas

La atencion calcula `Q`, `K` y `V` con una proyeccion lineal compartida y despues
divide la representacion en cabezas. La matriz de atencion usa el producto
`Q @ K.T`; se divide por `sqrt(head_dim)` para evitar logits demasiado grandes,
que saturarian la softmax y producirian gradientes poco utiles.

La softmax transforma los logits de atencion en pesos positivos que suman 1, de
modo que cada vector de contexto es una combinacion ponderada de los `values`.

En generacion se usa `causal=True` para impedir que una posicion mire tokens
futuros. En NER se usa `causal=False` porque la etiqueta de una palabra puede
depender tanto del contexto izquierdo como del derecho.

El alineamiento palabra -> BPE replica la etiqueta BIO de cada palabra sobre sus
subtokens. Si una palabra `B-LOC` se divide en varias piezas, la primera conserva
`B-LOC` y las siguientes reciben `I-LOC`.

## Reproducibilidad

Los entrenamientos fijan `seed=42` por defecto para Python, NumPy y PyTorch.
Cada checkpoint guarda la semilla, la configuracion completa y las metricas del
ultimo epoch. El dispositivo por defecto es CPU para que la practica sea
reproducible sin GPU.

## Limitaciones CPU

El corpus y el modelo son pequenos por diseno. En CPU conviene empezar con
`context_size=64`, `n_layers=3` y `epochs=5` para validar el flujo completo. La
configuracion base mejora capacidad, pero tarda mas y puede requerir paciencia.

## Experimentos recomendados

El informe debe comparar:

- `vocab_size`: 200 vs 400 vs 600
- `context_size`: 64 vs 128
- `n_layers`: 2 vs 4
- `dropout`: 0.0 vs 0.1
- `temperature`: 0.5 vs 0.8 vs 1.2

Para cada experimento se recomienda guardar perdida de entrenamiento,
perdida de validacion, ejemplos generados y una interpretacion breve.

## Formato, build y release

```bash
uv format --check
uv build
```

El wheel esperado para la release `p5v1.0` es:

```text
dist/fdi_pln_2608_p5-1.0.0-py3-none-any.whl
```

## Entrega final

La entrega debe incluir:

- `README.md`
- `reports/informe_2608.md`
- `checkpoints/p5_causal_2608.pth`
- `checkpoints/p5_ner_2608.pth`
- `dist/fdi_pln_2608_p5-1.0.0-py3-none-any.whl`
