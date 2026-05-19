# Informe P5 - Grupo 2608

## 1. Introduccion

Objetivo de la practica, corpus utilizado, alcance del mini LLM y relacion con
la tarea NER.

## 2. Tokenizacion BPE

Explicar vocabulario inicial, conteo de pares frecuentes, merges, codificacion y
decodificacion. Incluir ejemplos reales del corpus y comentar el efecto de
`vocab_size`:

- 200
- 400
- 600

El tokenizador parte de caracteres individuales observados en el corpus. En cada
iteracion cuenta pares adyacentes, selecciona el par mas frecuente y crea un
nuevo token que representa la fusion. Al codificar un texto nuevo, se aplican
los merges aprendidos en el mismo orden. Esto permite representar palabras
frecuentes con pocos tokens y palabras raras como composiciones de piezas mas
pequenas.

Comando para obtener ejemplos reproducibles:

```bash
uv run fdi-pln-2608-p5 analyze-bpe \
  --weights checkpoints/p5_causal_2608.pth \
  --text "Alice went to Wonderland"
```

Tabla para pegar analisis:

| Texto | Caracteres | Tokens | Caracteres/token | Segmentacion |
| --- | ---: | ---: | ---: | --- |
|  |  |  |  |  |

## 3. Atencion escalada

Describir `Q`, `K`, `V`, producto escalar, escalado por `sqrt(head_dim)`,
softmax, dropout y vector contexto. Explicar por que el escalado evita
saturacion de softmax y por que la softmax convierte puntuaciones en pesos de
atencion interpretables.

Para cada posicion, el modelo calcula queries, keys y values mediante
proyecciones lineales. El producto `Q @ K.T` mide compatibilidad entre cada
query y cada key. Si la dimension de cada cabeza es grande, la varianza de estos
productos escala con `head_dim`; dividir por `sqrt(head_dim)` mantiene los
logits en un rango mas estable y evita que la softmax se vuelva demasiado
puntiaguda al principio del entrenamiento.

La softmax convierte logits arbitrarios en una distribucion: todos los pesos son
positivos y suman 1. Asi, el vector de contexto se interpreta como una media
ponderada de los values. Dropout sobre la matriz de atencion reduce dependencia
excesiva de conexiones concretas.

La atencion causal enmascara posiciones futuras con `-inf` antes de la softmax,
por lo que en generacion cada token solo usa el pasado. La atencion
bidireccional no aplica esa mascara y permite usar contexto izquierdo y derecho;
por eso se usa en NER, donde la etiqueta de una palabra puede depender de lo que
aparece despues.

## 4. Transformer

Describir embeddings de token y posicion, multi-head attention, residuales,
LayerNorm pre-norm, feed-forward con GELU, dropout y batching.

Comparar:

- `context_size`: 64 vs 128
- `n_layers`: 2 vs 4
- `dropout`: 0.0 vs 0.1

## 5. LLM causal

Explicar modelado autoregresivo, desplazamiento `x -> y`, cross entropy,
mascara causal y generacion token a token. Justificar `causal=True` porque el
modelo no debe ver tokens futuros durante entrenamiento ni inferencia.

Incluir ejemplos con:

- `temperature=0.5`
- `temperature=0.8`
- `temperature=1.2`

## 6. NER

Explicar BIO tagging, etiquetas usadas (`O`, `B-PER`, `I-PER`, `B-LOC`,
`I-LOC`), cabeza de clasificacion por token y uso de `causal=False` para acceder
al contexto completo de la frase.

El modelo NER reutiliza embeddings, posiciones y bloques Transformer del modelo
causal preentrenado. La cabeza `lm_head` se sustituye por una capa lineal que
predice una etiqueta BIO por subtoken. Para alinear palabras con BPE, la primera
pieza de una palabra etiquetada como `B-X` mantiene `B-X` y las piezas
siguientes reciben `I-X`. Las posiciones de padding usan `ignore_index=-100`
para no afectar a la perdida ni a las metricas.

Metricas preparadas:

- token accuracy
- token precision, recall y F1, excluyendo la clase `O`
- entity precision, recall y F1, con coincidencia exacta BIO

Comando:

```bash
uv run fdi-pln-2608-p5 eval-ner \
  --weights checkpoints/p5_ner_2608.pth \
  --data data/ner/final.conll
```

Tabla para resultados NER:

| Modelo | Token acc | Token P | Token R | Token F1 | Entity P | Entity R | Entity F1 | Observaciones |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
|  |  |  |  |  |  |  |  |  |

## 7. Corpus y anotacion

Describir corpus preentrega, criterios de anotacion, metadatos, desacuerdos,
kappa de Cohen y decisiones finales. Indicar ejemplos ambiguos y como se
resolvieron.

## 8. Experimentos

Tabla sugerida:

| Experimento | vocab | context | d_model | heads | layers | dropout | epochs | train loss | val loss | observaciones |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | 400 | 128 | 128 | 4 | 4 | 0.1 | 8 |  |  |  |
| vocab pequeno | 200 | 128 | 128 | 4 | 4 | 0.1 | 8 |  |  |  |
| vocab grande | 600 | 128 | 128 | 4 | 4 | 0.1 | 8 |  |  |  |
| contexto corto | 400 | 64 | 128 | 4 | 4 | 0.1 | 8 |  |  |  |
| menos capas | 400 | 128 | 128 | 4 | 2 | 0.1 | 8 |  |  |  |
| sin dropout | 400 | 128 | 128 | 4 | 4 | 0.0 | 8 |  |  |  |

Generar tabla de muestras:

```bash
uv run fdi-pln-2608-p5 experiment-generate \
  --weights checkpoints/p5_causal_2608.pth \
  --prompt "Alice was" \
  --out reports/generation_experiments.md
```

Espacio para pegar resumen de `reports/generation_experiments.md`:

| Temperature | Top-k | Calidad | Repeticion | Coherencia | Observaciones |
| ---: | ---: | --- | --- | --- | --- |
| 0.5 | 10 |  |  |  |  |
| 0.5 | 20 |  |  |  |  |
| 0.5 | 50 |  |  |  |  |
| 0.8 | 10 |  |  |  |  |
| 0.8 | 20 |  |  |  |  |
| 0.8 | 50 |  |  |  |  |
| 1.2 | 10 |  |  |  |  |
| 1.2 | 20 |  |  |  |  |
| 1.2 | 50 |  |  |  |  |

## 9. Resultados

Interpretar perdidas, ejemplos generados y entidades detectadas. Comparar
calidad, coherencia local, repeticion, errores BIO y sensibilidad a la
temperatura.

## 10. Errores y limitaciones

Comentar limites por CPU, tamano de corpus, vocabulario pequeno, anotacion NER
reducida, entidades ambiguas, errores por subtokenizacion y dependencia de la
semilla.

## 11. Conclusiones

Resumir que se ha aprendido sobre BPE, atencion, Transformers, generacion
causal, transferencia de backbone y NER bidireccional. Incluir posibles mejoras:
mas corpus, mas anotaciones, metricas por entidad, early stopping y busqueda de
hiperparametros mas sistematica.
