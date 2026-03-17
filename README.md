# Procesamiento de Lenguaje Natural (PLN) 2026 🤖📚
**Grupo G08 - Grado en Ingeniería de Datos e Inteligencia Artificial (UCM)**

Este repositorio contiene el conjunto de prácticas desarrolladas durante el curso, que abarcan desde la creación de agentes inteligentes y síntesis de voz, hasta criptoanálisis de codificaciones propias y sistemas de recuperación de información.

# 👥 Equipo de Desarrollo

- **Carlota Salazar Martín**
- **Marina Triviño de las Heras**

# 📂 Estructura del Proyecto

El repositorio está organizado por prácticas, cada una con su propia lógica, dependencias y documentación específica:

### [P1] Trading Bot - Agente de Negociación ⚖️

Localizado en `/p1-g08`. Un agente capaz de interactuar con un servidor de trueque mediante lenguaje natural.

- **Core**: Integración con **LLM (Ollama/Qwen3)** para interpretar ofertas.
- **Lógica**: Gestión de inventario, detección de recursos faltantes/sobrantes y sistema anti-spam.
- **Tecnologías**: Python, `uv`, `client-server` communication.

### [P2] Fonética y Síntesis por Concatenación 🎙️

Localizado en `/p2-g08`. Análisis de la señal de voz y reconstrucción manual de pangramas.

- **Proceso**: Grabación, segmentación en **Praat** y ensamblaje mediante crossfading.
- **Análisis**: Estudio técnico sobre la coarticulación, prosodia y discontinuidad espectral en la síntesis del habla.

### [P3] Detectives de Criptoglifos (PLNCG26) 🔍

Localizado en `/p3-g08_C` y `/p3-g08_M`. Ingeniería inversa de un sistema de codificación binario desconocido.

- **Reto**: Análisis de frecuencias y descubrimiento de modificadores de bytes (acentos, mayúsculas, tildes).
- **Script**: Herramienta CLI para `encode`, `decode` y `detect` de archivos en formato PLNCG26.
- **Algoritmo**: Basado en desplazamientos constantes y gestión de diacríticos por bytes adyacentes.

### [P4] Quijote App - Recuperación de Información 📖

Localizado en `/p4-g08`. Buscador profesional de pasajes literales de El Quijote.

- **Funcionalidad**: Indexación de corpus HTML/ZIP, normalización de texto y ranking de relevancia.
- **Interfaz**: Modo interactivo en terminal con resaltado de coincidencias y filtrado por capítulos.
- **Tecnologías**: `BeautifulSoup4`, `Typer`, `joblib` para caché y `pytest`.

# 🛠️ Configuración Global

Este proyecto utiliza uv como gestor de paquetes y entornos de Python para garantizar la reproducibilidad.

**Requisitos previos**

- Tener instalado uv.
- (Para P1) Tener instalado Ollama con el modelo `qwen3-vl:8b`.

**Instalación rápida**
```
# Clonar el repositorio
git clone https://github.com/maritriv/fdi-pln2608.git
cd fdi-pln2608

# Sincronizar dependencias generales
uv sync
```

------

# 🚀 Cómo ejecutar las prácticas

Cada carpeta contiene su propio manual de uso, pero aquí tienes los comandos principales:

- **P1 (Bot)**: `export FDI_PLN__BUTLER_ADDRESS="http://..." && uv run fdi-pln-2608-p1`

- **P3 (Cripto)**: `uv run p3-g08_C/fdi-pln-2608-p3.py decode archivo.bin`

- **P4 (Buscador)**: `python -m quijote_app interactive

------

# 📝 Notas de Entrega

- La **Práctica 3** presenta dos implementaciones individuales (`_C` y `_M) debido a la metodología de investigación seguida en el laboratorio.
- La **Práctica 4** incluye una suite de tests completa para asegurar la integridad de la normalización y el motor de búsqueda.
