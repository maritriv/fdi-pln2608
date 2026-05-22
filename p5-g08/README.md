# P5: Toves, Borogoves & Mome Raths

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![uv](https://img.shields.io/badge/build-uv-4b32c3)
![Torch](https://img.shields.io/badge/dependency-torch-ee4c2c)
![Entrega](https://img.shields.io/badge/release-p5v1.0-success)

## Descripción general

El significado del lenguaje se basa en el contexto. Para comprender de verdad cómo una máquina es capaz de asimilarlo y procesarlo, no nos servía quedarnos en la superficie interactuando con modelos ya hechos; necesitábamos construir su arquitectura desde cero.

Este proyecto es un acercamiento a la anatomía de un modelo de lenguaje. Hemos implementado desde cero un pequeño modelo basado en la arquitectura Transformer, construyendo y comprendiendo sus piezas fundamentales: desde la tokenización BPE propia y los embeddings, hasta la atención escalada (*scaled dot-product self-attention*).

Posteriormente, dimos un paso más: adaptamos este modelo generativo a una tarea de Reconocimiento de Entidades Nombradas (NER) mediante transferencia de conocimiento (*Transfer Learning*). Aunque nuestro modelo es modesto y ha sido entrenado íntegramente en CPU, implementa de principio a fin el pipeline real de un sistema moderno de Procesamiento del Lenguaje Natural.

---

## Decisiones de diseño y arquitectura

Para mantener el código organizado, reproducible y fácil de usar, empaquetamos el proyecto siguiendo estándares profesionales de ingeniería de software (gestionado con `uv` y estructurado en módulos).

A nivel de red neuronal, diseñamos la arquitectura para que pudiera evolucionar y soportar dos grandes tareas cambiando solo un par de piezas clave:

1.  **Modelo Causal (Generación):** Un Transformer autorregresivo con atención enmascarada (*causal mask*). El modelo aprende a predecir el siguiente token leyendo exclusivamente el pasado. Entrenamos este modelo sobre textos clásicos de Lewis Carroll (fragmentos de *Alice in Wonderland* y *Through the Looking-Glass*), buscando que interiorizara la estructura de los diálogos y el vocabulario.

3.  **Modelo NER (Clasificación):** Aquí "reciclamos" el conocimiento. Tomamos el *backbone* del modelo causal (sus embeddings y capas Transformer), eliminamos la máscara causal para permitir **atención bidireccional** (el contexto posterior es vital para clasificar entidades), y cambiamos la cabeza de generación por una capa lineal de clasificación BIO. Usamos el corpus que nosotras mismas anotamos en la preentrega (junto con el resto del Grupo 1).


> Sobre nuestras etiquetas NER (Clasificación en formato BIO adaptado): En el corpus, hay que clasificar cada palabra con alguna de las siguientes etiquetas:
> - 'o' - si no es personaje
> - 'pi' - si es la primera palabra de un personaje
> - 'pc' - si es la continuación del personaje
> - 'li' - si es el inicio de un lugar
> - 'lc' - si es la continuación de un lugar
> - Atención: los espacios son 'o' excepto que estén entre medias de un personaje / lugar

---

## El pipeline paso a paso

Para nosotras era vital entender el orden lógico en el que se entrena y evalúa un sistema de estas características. Estructuramos nuestro flujo de trabajo en fases estrictamente secuenciales, donde la salida de cada paso alimentaba al siguiente:

**1. Alfabetización (Tokenización BPE)**
*   Entrenamiento de nuestro propio tokenizador BPE sobre el corpus literario crudo. Sin este "diccionario" base, la red no tendría forma de ingerir texto.

**2. Comprensión general (Pre-entrenamiento Causal)**
*   Entrenamiento del Transformer causal autoregresivo.
*   Pruebas empíricas de generación de texto para comprobar que el modelo empezaba a hablar como Lewis Carroll, y guardado del punto de control (`checkpoints/p5_causal_2608.pth`).

**3. Preparación de nuestro terreno (Datos NER)**
*   Integración del corpus que anotamos manualmente en la preentrega junto con el resto del Grupo 1.
*   Conversión automática de esos JSONs crudos al estándar tabular CoNLL/BIO, dejándolos listos para que la red los consuma.

**4. Especialización (Transfer Learning para NER)**
*   Reutilización del *backbone* (pesos y capas) del modelo Transformer causal.
*   Entrenamiento exclusivo de la nueva cabeza de clasificación BIO usando nuestros datos CoNLL.

**5. Validación y Puesta en Producción**
*   Evaluación cuantitativa calculando tanto las métricas *token-level* (muy altas por la abundancia de la clase 'O') como las *entity-level* (más honestas respecto al rendimiento real).
*   Inferencia final probando el modelo completo para extraer entidades sobre texto libre o arbitrario.

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
