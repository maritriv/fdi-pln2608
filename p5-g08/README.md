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

Toda la práctica se ejecuta desde un único CLI:

```bash
uv run fdi-pln-2608-p5
```

Al lanzar el comando principal sin subcomandos se abre un menú interactivo para
probar las funciones principales con valores por defecto: generación de texto,
NER, evaluación del modelo NER y análisis BPE.

La ayuda completa y los comandos disponibles pueden consultarse mediante:

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

Desde la raíz del repositorio:

```bash
cd p5-g08
uv sync
uv run fdi-pln-2608-p5
```

El primer comando instala las dependencias declaradas en `pyproject.toml`. El
segundo abre la aplicación interactiva de la práctica.

---

## Uso del programa

Toda la práctica se maneja desde un único ejecutable. Puede usarse de dos
formas: con un menú guiado o con subcomandos directos.

## Uso interactivo

Para abrir la aplicación:

```bash
cd p5-g08
uv run fdi-pln-2608-p5
```

Al ejecutarlo sin argumentos aparece un menú visual en la terminal:

```text
P5 · Transformer + NER
Grupo 2608

¿Qué quieres hacer?

  [1] Generar texto
  [2] Detectar entidades NER en un fichero
  [3] Evaluar el modelo NER
  [4] Analizar tokenización BPE
  [5] Ver comandos disponibles
  [0] Salir
```

El uso normal es escribir el número de la opción y pulsar Enter. Cada flujo pide
solo los parámetros necesarios y muestra el valor por defecto entre corchetes.
Si pulsas Enter sin escribir nada, se usa ese valor.

Defaults principales:

| Opción | Valores por defecto |
| --- | --- |
| Generar texto | `checkpoints/p5_causal_2608.pth`, prompt `Alice was`, 80 tokens, top-k 20, temperature 1.0 |
| Detectar entidades | `checkpoints/p5_ner_2608.pth`, `examples/text.txt` |
| Evaluar NER | `checkpoints/p5_ner_2608.pth`, `data/ner/final.conll` |
| Analizar BPE | `checkpoints/p5_causal_2608.pth`, texto `Alice went to Wonderland` |

Después de cada operación, el programa muestra el resultado en una sección
separada y espera a que pulses Enter para volver al menú. Para salir, elige
`0`.

Ejemplo de sesión:

```text
uv run fdi-pln-2608-p5
Selección: 1
Checkpoint [checkpoints/p5_causal_2608.pth]:
Prompt [Alice was]:
Max new tokens [80]: 20
Top-k [20]:
Temperature [1.0]:
```

## Comandos directos

El modo interactivo es cómodo para enseñar la práctica, pero los comandos
directos siguen siendo la forma más reproducible de lanzar experimentos:

```bash
uv run fdi-pln-2608-p5 generate --weights checkpoints/p5_causal_2608.pth --prompt "Alice was"
uv run fdi-pln-2608-p5 ner --weights checkpoints/p5_ner_2608.pth --file examples/text.txt
uv run fdi-pln-2608-p5 eval-ner --weights checkpoints/p5_ner_2608.pth --data data/ner/final.conll
```

Para ver todos los parámetros:

```bash
uv run fdi-pln-2608-p5 --help
```

Comandos principales:

| Comando | Uso |
| --- | --- |
| `train-causal` | Entrena el modelo causal de generación. |
| `generate` | Genera texto a partir de un prompt. |
| `prepare-ner-data` | Convierte `merged.json` a CoNLL/BIO. |
| `train-ner` | Entrena el modelo NER desde el checkpoint causal. |
| `eval-ner` | Evalúa el checkpoint NER sobre un dataset CoNLL. |
| `ner` | Detecta entidades nombradas en un fichero o texto. |
| `analyze-bpe` | Muestra la segmentación BPE de un texto. |
| `experiment-generate` | Genera una tabla de experimentos de generación. |

---


## Generación de texto

```bash
uv run fdi-pln-2608-p5 generate --weights checkpoints/p5_causal_2608.pth --prompt "Alice was" --max-new-tokens 80 --top-k 20
```

Parámetros principales:

- `--weights`: checkpoint causal que se quiere cargar.
- `--prompt`: texto inicial proporcionado por el usuario.
- `--max-new-tokens`: número máximo de tokens generados.
- `--top-k`: restringe el muestreo a los tokens más probables.
- `--temperature`: controla la aleatoriedad de la salida.

---

## Entrenamiento causal

```bash
uv run fdi-pln-2608-p5 train-causal \
  --corpus resources \
  --output checkpoints/p5_causal_2608.pth
```

Este comando entrena el modelo de lenguaje causal sobre los textos de `resources/` y guarda un checkpoint autocontenido.

---

## Preparación de datos NER

```bash
uv run fdi-pln-2608-p5 prepare-ner-data \
  --input ../pre-entrega_2601/merged.json \
  --output data/ner/final.conll
```

Este paso convierte el corpus fusionado de la preentrega a formato CoNLL/BIO, que es el formato que consume `train-ner`.

---

## Entrenamiento NER

```bash
uv run fdi-pln-2608-p5 train-ner --data data/ner/final.conll --causal-weights checkpoints/p5_causal_2608.pth --output checkpoints/p5_ner_2608.pth
```

Este comando carga el modelo causal indicado con `--causal-weights`, reutiliza su backbone Transformer y guarda el checkpoint NER en `--output`.

---

## Evaluación NER

```bash
uv run fdi-pln-2608-p5 eval-ner --weights checkpoints/p5_ner_2608.pth --data data/ner/final.conll
```

La evaluación imprime métricas token-level y entity-level, y guarda un resumen en `reports/ner_metrics_2608.json`.

---

## Inferencia NER

```bash
uv run fdi-pln-2608-p5 ner --weights checkpoints/p5_ner_2608.pth --file examples/text.txt
```

También puede usarse texto directo:

```bash
uv run fdi-pln-2608-p5 ner \
  --weights checkpoints/p5_ner_2608.pth \
  --text "Alice met the Queen in Wonderland"
```

La salida es una lista tabulada con la entidad detectada y su tipo (`PER` o `LOC`).

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
