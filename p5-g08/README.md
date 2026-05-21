# P5: Toves, Borogoves & Mome Raths

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![uv](https://img.shields.io/badge/build-uv-4b32c3)
![Torch](https://img.shields.io/badge/dependency-torch-ee4c2c)
![Entrega](https://img.shields.io/badge/release-p5v1.0-success)

## Descripción general

Este proyecto implementa un pequeño modelo de lenguaje basado en la arquitectura Transformer y una tarea de reconocimiento de entidades nombradas (NER) construida sobre el mismo backbone.

La práctica se desarrolló con el objetivo de comprender e implementar manualmente los componentes principales de un pipeline moderno de NLP:

- tokenización BPE;
- embeddings y codificación posicional;
- scaled self-attention;
- generación autoregresiva;
- transferencia de pesos;
- y adaptación de un backbone Transformer a tareas NER.

El modelo causal se entrenó sobre fragmentos de *Alice in Wonderland* y *Through the Looking-Glass*, mientras que la parte NER reutiliza el corpus anotado manualmente durante la preentrega del grupo 1.

Aunque se trata de modelos relativamente pequeños y entrenados únicamente en CPU, el sistema completo funciona de forma reproducible y permite entrenar, generar texto, evaluar métricas e inferir entidades desde línea de comandos.

---

# Flujo general del proyecto

El pipeline seguido durante la práctica fue el siguiente:

1. entrenamiento del tokenizador BPE sobre el corpus literario;
2. entrenamiento del Transformer causal autoregresivo;
3. generación y guardado de checkpoints;
4. integración del corpus anotado manualmente durante la preentrega;
5. conversión automática a formato CoNLL/BIO;
6. reutilización del backbone Transformer para NER;
7. entrenamiento de la cabeza de clasificación BIO;
8. evaluación de métricas token-level y entity-level;
9. inferencia final sobre texto arbitrario.

Toda la práctica puede ejecutarse desde un único CLI:

```bash
uv run fdi-pln-2608-p5 --help
```

---

# Estructura del proyecto

```text
p5-g08/
├── checkpoints/                     # Checkpoints entrenados del modelo causal y NER
│   ├── p5_causal_2608.pth
│   └── p5_ner_2608.pth
│
├── data/                            # Datos integrados para la tarea NER
│   └── ner/
│       ├── merged.json              # Corpus fusionado procedente de la preentrega
│       └── final.conll              # Dataset convertido a formato BIO/CoNLL
│
├── examples/                        # Ejemplos de texto para pruebas e inferencia
│   └── text.txt
│
├── reports/                         # Informes y métricas generadas durante evaluación
│   ├── informe_2608.md
│   └── ner_metrics_2608.json
│
├── resources/                       # Corpus literario utilizado para generación causal
│   ├── alice_in_wonderland.txt
│   └── looking_glass.txt
│
├── src/                             # Código fuente principal del proyecto
│   └── fdi_pln_2608_p5/
│       ├── modules/                 # Componentes internos del Transformer y NER
│       ├── tokenizer.py             # Implementación del tokenizador BPE
│       ├── attention.py             # Scaled multi-head self-attention
│       ├── transformer.py           # Bloques Transformer y arquitectura base
│       ├── causal_llm.py            # Modelo causal autoregresivo
│       ├── ner_llm.py               # Modelo NER reutilizando el backbone
│       ├── prepare_ner_data.py      # Conversión automática a formato BIO/CoNLL
│       ├── checkpoint.py            # Guardado y carga de checkpoints
│       └── main.py                  # CLI principal del proyecto
│
├── dist/                            # Wheel generado para la entrega final
├── README.md                        # Documentación principal del proyecto
├── pyproject.toml                   # Configuración del paquete Python
├── uv.lock                          # Lockfile reproducible de dependencias
└── .gitignore                       # Exclusión de artefactos y cachés
```

---

# Arquitectura

El proyecto está dividido en módulos relativamente independientes para mantener el código organizado y reutilizable.

## Tokenización

`tokenizer.py`

Implementa un tokenizador BPE sencillo entrenado sobre los textos de Lewis Carroll. El vocabulario se construye fusionando pares frecuentes de caracteres.

---

## Atención y Transformer

`attention.py`
`transformer.py`

Implementan:

- multi-head self-attention;
- máscara causal opcional;
- residual connections;
- LayerNorm;
- feed-forward con GELU;
- y embeddings posicionales.

El modelo causal utiliza atención enmascarada (`causal=True`), mientras que NER utiliza atención bidireccional (`causal=False`).

---

## Modelo causal

`causal_llm.py`

Implementa el modelo autoregresivo encargado de generar texto token a token.

Se reutiliza posteriormente como backbone para la tarea NER.

---

## Modelo NER

`ner_llm.py`

Sustituye la cabeza de lenguaje del Transformer por una capa BIO de clasificación de entidades.

El modelo reutiliza embeddings y bloques Transformer aprendidos previamente durante el entrenamiento causal.

---

## Checkpoints

`checkpoint.py`

Los checkpoints son autocontenidos e incluyen:

```python
{
    "model_state_dict": ...,
    "tokenizer": ...,
    "config": ...,
    "metrics": ...,
    "seed": 42,
}
```

Esto permite cargar completamente el modelo sin depender de configuraciones externas.

---

# Corpus utilizado

## Generación causal

Para el entrenamiento del modelo causal se utilizaron:

- `resources/alice_in_wonderland.txt`
- `resources/looking_glass.txt`

Se eligieron estos textos por contener:

- abundantes diálogos;
- personajes recurrentes;
- vocabulario relativamente variado;
- y un estilo literario fácilmente reconocible.

---

## Corpus NER

La parte NER reutiliza el corpus anotado manualmente durante la preentrega.

Fuente integrada:

```text
data/ner/merged.json
```

Conversión automática a CoNLL/BIO:

```bash
uv run fdi-pln-2608-p5 prepare-ner-data --input ../pre-entrega_2601/merged.json --output data/ner/final.conll
```

Resumen del corpus:

| Medida | Valor |
| --- | ---: |
| Frases fusionadas | 59 |
| Tokens anotados | 5750 |
| Tokens CoNLL | 3263 |
| Tokens de entidad | 194 |
| Kappa medio | 0.835 |
| Acuerdo medio | 98% |

Mapeo de etiquetas:

| Preentrega | BIO |
| --- | --- |
| `o` | `O` |
| `pi` | `B-PER` |
| `pc` | `I-PER` |
| `li` | `B-LOC` |
| `lc` | `I-LOC` |

---

# Quickstart

Instalación y ayuda general:

```bash
uv sync
uv run fdi-pln-2608-p5 --help
```

---

## Generación de texto

```bash
uv run fdi-pln-2608-p5 generate --weights checkpoints/p5_causal_2608.pth --prompt "Alice was" --max-new-tokens 80 --top-k 20
```

---

## Entrenamiento NER

```bash
uv run fdi-pln-2608-p5 train-ner --data data/ner/final.conll --causal-weights checkpoints/p5_causal_2608.pth --output checkpoints/p5_ner_2608.pth
```

---

## Evaluación NER

```bash
uv run fdi-pln-2608-p5 eval-ner --weights checkpoints/p5_ner_2608.pth --data data/ner/final.conll
```

---

## Inferencia NER

```bash
uv run fdi-pln-2608-p5 ner --weights checkpoints/p5_ner_2608.pth --file examples/text.txt
```

---

# Resultados obtenidos

## Generación causal

El modelo consigue capturar parcialmente el estilo del corpus original:

- aparecen diálogos;
- puntuación similar;
- y nombres propios característicos de Lewis Carroll.

Sin embargo, debido al tamaño reducido del dataset, todavía aparecen:

- palabras deformadas;
- repeticiones;
- y pérdida de coherencia en textos largos.

Ejemplo real:

```text
Alice was a worstanding large carriled it, and had put it: she stood out of
we frightened so day.
```

---

## Resultados NER

Métricas principales:

| Métrica | Valor |
| --- | ---: |
| Token accuracy | 0.8881 |
| Token precision | 0.2666 |
| Token recall | 0.7053 |
| Token F1 | 0.3869 |
| Entity precision | 0.1132 |
| Entity recall | 0.5909 |
| Entity F1 | 0.1901 |

Desglose entity-level:

| Tipo | F1 |
| --- | ---: |
| LOC | 0.1840 |
| PER | 0.1919 |

El modelo logra detectar bastantes entidades relevantes del corpus, especialmente personajes frecuentes, aunque todavía aparecen falsos positivos y confusiones entre tipos.

Dado el tamaño del corpus anotado y el fuerte desbalance hacia la clase `O`, estos resultados eran relativamente esperables.

---

# Reproducibilidad

Validación recomendada antes de entregar:

```bash
uv sync
uv format --check
uv run fdi-pln-2608-p5 --help

uv run fdi-pln-2608-p5 generate --weights checkpoints/p5_causal_2608.pth --prompt "Alice was"

uv run fdi-pln-2608-p5 eval-ner --weights checkpoints/p5_ner_2608.pth --data data/ner/final.conll

uv build
```

Wheel generado:

```text
dist/fdi_pln_2608_p5-1.0-py3-none-any.whl
```

---

# Limitaciones

Las principales limitaciones encontradas fueron:

- corpus relativamente pequeño;
- fuerte desbalance entre entidades y tokens `O`;
- entrenamiento únicamente en CPU;
- y vocabulario limitado.

En generación causal esto produce:

- repeticiones;
- deformaciones léxicas;
- y pérdida de coherencia en secuencias largas.

En NER aparecen:

- falsos positivos;
- errores de delimitación;
- y confusión entre PER y LOC.

Aun así, el pipeline completo funciona correctamente y permite entrenar, evaluar e inferir entidades de forma reproducible.

---

# Integrantes

Estudiantes del grado de Ingeniería de Datos e Inteligencia Artificial de la Universidad Complutense de Madrid.

- Marina Triviño de las Heras
- Carlota Salazar Martín