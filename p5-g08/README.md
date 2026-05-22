# P5: Toves, Borogoves & Mome Raths

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![uv](https://img.shields.io/badge/build-uv-4b32c3)
![Torch](https://img.shields.io/badge/dependency-torch-ee4c2c)
![Entrega](https://img.shields.io/badge/release-p5v1.0-success)

## Descripción general

El significado del lenguaje se basa en el contexto. Hoy en día es muy fácil usar un gran modelo de lenguaje (LLM) como una caja negra a través de una API, pero para entender realmente el culpable de la revolución actual de la IA, sentíamos que necesitábamos ensuciarnos las manos.

Este proyecto es un acercamiento a la anatomía de un modelo de lenguaje. Hemos implementado desde cero un pequeño modelo basado en la arquitectura Transformer, construyendo y comprendiendo sus piezas fundamentales: desde la tokenización BPE propia y los embeddings, hasta la atención escalada (*scaled dot-product self-attention*).

Posteriormente, dimos un paso más: adaptamos este modelo generativo a una tarea de Reconocimiento de Entidades Nombradas (NER) mediante transferencia de conocimiento (*Transfer Learning*). Aunque nuestro modelo es modesto y ha sido entrenado íntegramente en CPU, implementa de principio a fin el pipeline real de un sistema moderno de Procesamiento del Lenguaje Natural.

---

## Decisiones de diseño y arquitectura

Para mantener el código organizado, reproducible y fácil de usar, empaquetamos el proyecto siguiendo estándares profesionales de ingeniería de software (gestionado con `uv` y estructurado en módulos).

Nuestra arquitectura se divide en dos grandes fases de aprendizaje:

1.  **Modelo Causal (Generación):** Un Transformer autorregresivo con atención enmascarada (*causal mask*). El modelo aprende a predecir el siguiente token leyendo exclusivamente el pasado. Entrenamos este modelo sobre textos clásicos de Lewis Carroll (fragmentos de *Alice in Wonderland* y *Through the Looking-Glass*), buscando que interiorizara la estructura de los diálogos y el vocabulario.

2.  **Modelo NER (Clasificación):** Aquí "reciclamos" el conocimiento. Tomamos el *backbone* del modelo causal (sus embeddings y capas Transformer), eliminamos la máscara causal para permitir **atención bidireccional** (el contexto posterior es vital para clasificar entidades), y cambiamos la cabeza de generación por una capa de clasificación BIO. Usamos el corpus que nosotras mismas anotamos en la preentrega (junto con el resto del Grupo 1).


> Sobre nuestras etiquetas NER (Clasificación en formato BIO adaptado): En el corpus, hay que clasificar cada palabra con alguna de las siguientes etiquetas:
> - 'o' - si no es personaje
> - 'pi' - si es la primera palabra de un personaje
> - 'pc' - si es la continuación del personaje
> - 'li' - si es el inicio de un lugar
> - 'lc' - si es la continuación de un lugar
> - Atención: los espacios son 'o' excepto que estén entre medias de un personaje / lugar

---

## Estructura del repositorio

Hemos separado las responsabilidades lógicas para que cada componente pueda evolucionar (o ser evaluado) de forma independiente:

```text
p5-g08/
|-- checkpoints/                     # Modelos entrenados autocontenidos listos para inferencia
|-- data/ner/                        # Nuestro corpus NER convertido automáticamente a CoNLL
|-- examples/                        # Textos de prueba y prompts
|-- reports/                         # Informes, métricas y diario de experimentos
|-- resources/                       # Corpus literario original para pre-entrenamiento
|
|-- src/fdi_pln_2608_p5/
|   |-- cli.py                       # Entry point del wheel
|   |-- main.py                      # Import mínimo de la app Typer
|   |-- cli_app/                     # Interfaz visual de terminal (Typer + Rich)
|   |-- model/                       # Arquitectura base: attention, transformer y adaptación NER
|   |-- data/                        # Carga de datos, datasets y conversión de formatos
|   |-- training/                    # Bucles de entrenamiento
|   |-- evaluation/                  # Cálculo de métricas y análisis de errores
|   |-- generation/                  # Inferencia, autoregresión y predicción de etiquetas
|   |-- tokenizer.py                 # Nuestro tokenizador BPE artesanal
|   `-- checkpoint.py                # Lógica para empaquetar pesos, configs y tokenizador
|
|-- dist/                            # Wheel compilado de la entrega (.whl)
`-- pyproject.toml                   # Gestión de dependencias
```

---

## Instalación y Uso (Quickstart)

Queríamos que evaluar nuestro proyecto fuera una experiencia agradable. Por eso, además de los comandos tradicionales, construimos una interfaz CLI enriquecida visualmente. Todo se maneja desde un único ejecutable.

Para arrancar, instala las dependencias y lanza el programa:

```bash
cd p5-g08
uv sync
uv run fdi-pln-2608-p5
```

_El primer comando instala las dependencias declaradas en `pyproject.toml`. El segundo abre la aplicación interactiva de la práctica._

### El Menú Interactivo
Si ejecutas el comando anterior sin argumentos, te recibirá nuestro menú guiado. Lo diseñamos para que se puedan probar las funciones principales (generar texto, detectar entidades, evaluar métricas o analizar cómo segmenta el BPE) de forma muy intuitiva, usando valores por defecto sensatos si simplemente pulsas Enter.

### Comandos Directos para Reproducibilidad
Si prefieres lanzar experimentos o automatizar pruebas, mantenemos los subcomandos directos. Aquí mostramos los más relevantes:

**Generar texto:**

```Bash
uv run fdi-pln-2608-p5 generate --weights checkpoints/p5_causal_2608.pth --prompt "Alice was" --max-new-tokens 80 --temperature 0.8
```

**Predecir entidades en un archivo:**

```Bash
uv run fdi-pln-2608-p5 ner --weights checkpoints/p5_ner_2608.pth --file examples/text.txt
```

**Pipeline completo (si quisieras reentrenar todo):**
```Bash
uv run fdi-pln-2608-p5 train-causal --corpus resources --output checkpoints/p5_causal_2608.pth
uv run fdi-pln-2608-p5 prepare-ner-data --input ../pre-entrega_2601/merged.json --output data/ner/final.conll
uv run fdi-pln-2608-p5 train-ner --data data/ner/final.conll --causal-weights checkpoints/p5_causal_2608.pth --output checkpoints/p5_ner_2608.pth
```

Por último, la ayuda completa y los comandos disponibles pueden consultarse mediante:

```bash
uv run fdi-pln-2608-p5 --help
```

---

## Resultados y Rendimiento

Aunque el análisis exhaustivo se encuentra en la carpeta `reports/`, estos son los resultados principales de nuestros modelos entrenados en CPU:

### Generación Causal
El modelo logra interiorizar el estilo de Lewis Carroll, estructurando diálogos y usando nombres propios del corpus, aunque limitado por su tamaño. 
*   *Ejemplo generado (Temp 0.8):* "Alice was a pawn out of it, and she could not thought to herself, “if it would have a good must grine,"

### Reconocimiento de Entidades (NER)
A nivel de entidad, el modelo se enfrenta a un fuerte desbalance de clases (la mayoría del texto es `O`), lo que penaliza la precisión. Sin embargo, logra un **Recall del ~59%**, demostrando que es capaz de localizar a la mayoría de personajes principales de la obra.

| Tipo | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **LOC** | 0.1027 | 0.8824 | 0.1840 |
| **PER** | 0.1168 | 0.5376 | 0.1919 |

---

## Integrantes
Estudiantes del grado de Ingeniería de Datos e Inteligencia Artificial (Universidad Complutense de Madrid). Grupo 2608. Año 2026.

- **Marina Triviño de las Heras**
- **Carlota Salazar Martín**





----------------
# BORRAR A PARTIR DE AQUÍ
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
|-- checkpoints/                     # Checkpoints entrenados del modelo causal y NER
|   |-- p5_causal_2608.pth
|   `-- p5_ner_2608.pth
|
|-- data/ner/                        # Corpus NER integrado desde la preentrega
|   |-- merged.json
|   `-- final.conll
|
|-- examples/                        # Textos de ejemplo para inferencia
|-- reports/                         # Informes, métricas y experimentos
|-- resources/                       # Corpus literario usado para entrenamiento causal
|
|-- src/fdi_pln_2608_p5/
|   |-- cli.py                       # Entry point del wheel
|   |-- main.py                      # Import mínimo de la app Typer
|   |-- cli_app/                     # CLI Typer + Rich
|   |   |-- app.py                   # Definición de la aplicación
|   |   |-- commands.py              # Subcomandos directos
|   |   |-- interactive.py           # Menú interactivo
|   |   `-- render.py                # Paneles, tablas y prompts Rich
|   |
|   |-- model/                       # Arquitectura Transformer y NER
|   |   |-- attention.py
|   |   |-- transformer.py
|   |   `-- ner.py
|   |
|   |-- data/                        # Dataset causal y conversión NER a CoNLL
|   |   |-- dataset.py
|   |   `-- prepare_ner_data.py
|   |
|   |-- training/                    # Entrenamiento causal y NER
|   |-- evaluation/                  # Evaluación NER y análisis BPE
|   |-- generation/                  # Generación de texto e inferencia NER
|   |-- tokenizer.py                 # Tokenizador BPE propio
|   |-- checkpoint.py                # Guardado/carga de checkpoints
|   `-- utils.py                     # Semillas, dispositivo y utilidades comunes
|
|-- dist/                            # Wheel final de la entrega
|-- README.md
|-- pyproject.toml
|-- uv.lock
`-- .gitignore
```

La lógica está separada por responsabilidad: el CLI vive en `cli_app/`, el
modelo en `model/`, los datos en `data/`, el entrenamiento en `training/`, la
evaluación en `evaluation/` y la inferencia en `generation/`.

---

# Arquitectura

El proyecto está dividido en módulos relativamente independientes para mantener el código organizado y reutilizable.

## Tokenización

`tokenizer.py`

Implementa un tokenizador BPE sencillo entrenado sobre los textos de Lewis Carroll. El vocabulario se construye fusionando pares frecuentes de caracteres.

---

## Atención y Transformer

`src/fdi_pln_2608_p5/model/attention.py`
`src/fdi_pln_2608_p5/model/transformer.py`

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

`src/fdi_pln_2608_p5/model/transformer.py`

Implementa el modelo autoregresivo encargado de generar texto token a token.

Se reutiliza posteriormente como backbone para la tarea NER.

---

## Modelo NER

`src/fdi_pln_2608_p5/model/ner.py`

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
