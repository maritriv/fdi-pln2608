# Butler Trading Agent — PLN

## Descripción

Este proyecto implementa un agente autónomo para la práctica **“Los agentes de Butler”**
de la asignatura de Procesamiento del Lenguaje Natural.

El agente participa en un entorno multiagente de trueque de recursos,
interactuando mediante cartas y envío de paquetes a través del servidor Butler.

La arquitectura combina:

- reglas deterministas,
- parsing mediante expresiones regulares,
- extracción semántica mediante LLM local (Ollama),
- control de estado interno,
- mecanismos anti-spam y anti-duplicados.

El objetivo del agente es maximizar el progreso hacia sus objetivos
de recursos realizando intercambios beneficiosos con otros agentes.

---

## Estrategia híbrida

El sistema utiliza una estrategia híbrida compuesta por tres niveles
de interpretación de mensajes:

1. Parsing estructurado (`OFERTA_V1`)
2. Extracción mediante expresiones regulares
3. Interpretación semántica mediante LLM local

Esto permite:

- robustez frente a mensajes mal formados,
- compatibilidad entre agentes,
- tolerancia a lenguaje natural flexible,
- reducción de errores de interpretación.

El modelo LLM no toma decisiones estratégicas directamente;
su función se limita a extraer información estructurada
a partir de cartas escritas en lenguaje natural.

---

## Diseño modular

El sistema sigue una arquitectura modular en la que cada componente tiene una responsabilidad claramente definida. La comunicación con el servidor Butler, la lógica de negociación, el procesamiento lingüístico, la gestión del estado interno y la estrategia principal del agente se encuentran separados en módulos independientes.

Esta organización permite mantener el código más limpio y mantenible, facilita la depuración y reutilización de componentes, y hace posible ampliar o modificar el comportamiento del agente sin afectar al resto del sistema.
---

## Estructura general del repositorio

```text
fdi-pln2608/
├── pln/
│
├── api/                  # Comunicación con Butler
│   └── client.py
│
├── nlp/                  # Procesamiento de lenguaje natural
│   ├── llm.py
│   ├── normalize.py
│   └── parse.py
│
├── trading/              # Lógica de negociación
│   ├── logic.py
│   └── offers.py
│
├── config.py             # Configuración global
├── game.py               # Estrategia auxiliar
├── logger.py             # Sistema de logging
├── main.py               # Bucle principal
├── state.py              # Estado interno
│
├── pyproject.toml
├── uv.lock
└── README.md
```

## Arquitectura del proyecto

### `pln/api/`

Responsable de toda la comunicación HTTP con el servidor Butler.

#### `client.py`

- `get_info()` → obtiene recursos, objetivos, alias y buzón
- `get_gente()` → obtiene agentes conectados
- `enviar_carta()` → envía mensajes entre agentes
- `enviar_paquete()` → envía recursos
- `borrar_carta()` → elimina cartas procesadas

---

### `pln/nlp/`

Módulo encargado del procesamiento lingüístico.

#### `normalize.py`

Funciones de normalización y limpieza textual:

- normalización de recursos,
- normalización de texto libre,
- detección de alias propio,
- filtrado de mensajes del sistema.

#### `parse.py`

Parsing basado en reglas:

- `parse_oferta_v1()`
- `extraer_oferta_1x1_regex()`

#### `llm.py`

Interpretación flexible mediante LLM local.

La función:

```python
interpretar_carta_a_listas()
```

devuelve siempre:

```json
{
  "quiere": [...],
  "ofrece": [...]
}
```

---

### `pln/trading/`

Contiene la lógica principal de negociación.

#### `logic.py`

- evaluación de trueques,
- confirmación de intercambios,
- gestión de ofertas pendientes,
- validaciones de seguridad,
- control de duplicados.

#### `offers.py`

- cooldown anti-spam,
- registro de ofertas activas,
- generación de mensajes estructurados,
- selección de intercambios 1x1.

---

### `pln/state.py`

Estado interno persistente entre iteraciones del agente.

Contiene:

- `ULTIMO_ENVIO_A`
- `PAQUETES_ENVIADOS`
- `OFERTAS_PENDIENTES`

---

### `pln/game.py`

Funciones auxiliares de estrategia:

- cálculo de recursos sobrantes,
- cálculo de recursos faltantes,
- priorización de cartas,
- envío proactivo de ofertas.

---

### `pln/main.py`

Contiene el bucle principal del agente.

Gestiona:

- consulta del estado,
- procesamiento del buzón,
- evaluación de ofertas,
- envío de recursos,
- comportamiento probabilístico,
- control temporal del agente.

---

### `pln/config.py`

Configuración global del sistema:

- cooldowns,
- tiempos de espera,
- probabilidad de envío,
- URLs del servidor,
- parámetros del LLM.

---

### `pln/logger.py`

Sistema de logging centralizado:

```python
log("mensaje")
```

## Flujo de ejecución

### 1. Consulta del estado

El agente consulta periódicamente:

```text
GET /info
```

Obtiene:

- recursos actuales,
- objetivos,
- alias,
- buzón de mensajes.

---

### 2. Análisis estratégico

Se calculan:

- recursos sobrantes → posibles intercambios,
- recursos faltantes → objetivos prioritarios.

---

### 3. Procesamiento del buzón

El sistema prioriza:

1. confirmaciones pendientes,
2. nuevas ofertas.

La interpretación sigue este orden:

```text
OFERTA_V1 → Regex → LLM
```

Una oferta solo se acepta si:

- el recurso ofrecido pertenece a faltantes,
- el recurso solicitado pertenece a sobrantes.

---

### 4. Gestión de ofertas pendientes

Cuando otro agente acepta una oferta enviada previamente:

- se envía el recurso comprometido,
- se confirma el intercambio,
- se elimina el pendiente interno.

---

### 5. Ofertas proactivas

Con probabilidad configurable:

```python
PROB_ENVIAR_OFERTA
```

el agente:

- selecciona un destinatario aleatorio,
- respeta cooldown anti-spam,
- genera una propuesta 1x1,
- registra la oferta pendiente.

---

### 6. Espera aleatoria

Para evitar comportamiento determinista:

```python
sleep(random entre SLEEP_MIN y SLEEP_MAX)
```

---

## Robustez y seguridad

El sistema incorpora varias medidas de protección:

- prevención de autoenvíos,
- cooldown anti-spam,
- control de duplicados,
- limpieza automática de estado antiguo,
- tolerancia a errores de red,
- recuperación automática ante fallos temporales.

---

## Tecnologías utilizadas

- Python 3.12
- uv
- httpx
- Ollama
- JSON
- Expresiones regulares
- Arquitectura modular

---

## Variables de entorno

| Variable | Descripción |
|---|---|
| `FDI_PLN__BUTLER_ADDRESS` | Dirección del servidor Butler |
| `FDI_PLN__AGENTE` | Alias del agente en modo monopuesto |

---

## Instalación del entorno

### 1. Clonar repositorio

```bash
git clone https://github.com/maritriv/fdi-pln2608.git
cd fdi-pln2608
```

---

### 2. Instalar dependencias

Instalar `uv`:

```bash
pip install uv
```

Instalar dependencias del proyecto:

```bash
uv sync
```

---

### 3. Configurar Ollama

Descargar el modelo utilizado:

```bash
ollama pull qwen3-vl:8b
```

El sistema utiliza:

```text
http://localhost:11434/api/generate
```

---

## Ejecución

Definir la dirección del servidor Butler:

```bash
export FDI_PLN__BUTLER_ADDRESS="http://127.0.0.1:7719"
```

Ejecutar el agente:

```bash
uv run fdi-pln-2608-p1
```

Para detener el sistema:

```text
Ctrl + C
```

El agente captura correctamente la señal y finaliza de forma segura.

---

## Uso de LLM

El agente utiliza un modelo local ejecutado mediante Ollama
para interpretar mensajes ambiguos o escritos en lenguaje natural.

El LLM no toma decisiones estratégicas directamente.
Su función se limita a extraer información estructurada
a partir de cartas, mientras que la lógica de negociación
permanece controlada mediante reglas deterministas.

---

## Equipo de desarrollo

Proyecto desarrollado por estudiantes del Grado en Ingeniería de Datos e Inteligencia Artificial (UCM):

- Carlota Salazar Martín
- Marina Triviño de las Heras