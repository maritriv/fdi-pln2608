# Trading Bot

## 📖 Descripción

Proyecto de **Procesamiento del Lenguaje Natural**.

El objetivo es desarrollar un **agente basado en lenguaje natural** capaz de interactuar con un servidor de trueque mediante cartas y envío de recursos con otros agentes.

El agente es capaz de:

- Leer cartas del buzón
- Interpretar ofertas estructuradas y no estructuradas
- Aceptar intercambios cuando son estratégicamente favorables
- Enviar recursos automáticamente
- Confirmar intercambios
- Enviar ofertas proactivas con comportamiento parcialmente aleatorio
- Evitar spam mediante cooldown por destinatario

El sistema combina reglas deterministas, parsing por regex y extracción flexible mediante LLM (Ollama).

---

## 📁 Estructura general del repositorio
```
fdi-pln2608/
├── pln/
│ ├── api/ ← Comunicación con el servidor
│ │ └── client.py
│ │
│ ├── nlp/ ← Procesamiento de lenguaje
│ │ ├── llm.py
│ │ ├── normalize.py
│ │ └── parse.py
│ │
│ ├── trading/ ← Lógica de negociación
│ │ ├── logic.py
│ │ └── offers.py
│ │
│ ├── config.py ← Parámetros globales
│ ├── game.py ← Funciones auxiliares
│ ├── logger.py ← Sistema de logging
│ ├── main.py ← Bucle principal
│ └── state.py ← Estado interno del agente
│
├── pyproject.toml ← Configuración del proyecto (uv)
├── uv.lock ← Lockfile
└── README.md ← Documentación principal
```

---

## 📚 Índice detallado

### 📂 `pln/api/`

Responsable de la comunicación con el servidor.

**Archivo `client.py`:**

- `get_info()` → obtiene recursos, objetivos, alias y buzón  
- `get_gente()` → lista de jugadores  
- `enviar_carta()` → envía una carta  
- `enviar_paquete()` → envía recursos  
- `borrar_carta()` → elimina carta procesada  

---

### 📂 `pln/nlp/`

Procesamiento lingüístico.

**`normalize.py`**
- Normalización de nombres de recursos
- Normalización de texto libre
- Detección de alias propio
- Filtrado de cartas del sistema

**`parse.py`**
- `parse_oferta_v1()` → parseo estructurado
- `extraer_oferta_1x1_regex()` → extracción simple por regex

**`llm.py`**
- `interpretar_carta_a_listas()` → extracción flexible mediante LLM  
Devuelve siempre:

```json
{"quiere": [...], "ofrece": [...]}
```
---

### 📂 `pln/trading/`

Lógica del sistema de negociación.

**`logic.py`**

- `evaluar_y_ejecutar_trueque()` → decide si aceptar oferta  
- `es_carta_confirmacion_pendiente()` → detecta confirmaciones  
- `procesar_confirmacion_pendiente()` → completa intercambio pendiente  
- Funciones de confirmación de envío  

**`offers.py`**

- Gestión de cooldown anti-spam  
- Registro de ofertas pendientes  
- Generación de mensajes `[OFERTA_V1]`  
- Selección de intercambio 1x1  

---

### 📂 `pln/state.py`

Estado interno en memoria:

- `ULTIMO_ENVIO_A` → control anti-spam  
- `PAQUETES_ENVIADOS` → evitar duplicados  
- `OFERTAS_PENDIENTES` → seguimiento de intercambios abiertos  

---

### 📂 `pln/game.py`

Funciones auxiliares del flujo principal:

- Cálculo de recursos sobrantes  
- Cálculo de recursos faltantes  
- Selección de carta prioritaria  
- Envío de ofertas proactivas  

---

### 📂 `pln/config.py`

Parámetros configurables:

- Probabilidad de enviar oferta  
- Intervalo de espera entre iteraciones  
- Cooldown anti-spam  
- Dirección del servidor  

---

### 📂 `pln/logger.py`

Sistema de logging simple:

```python
log("mensaje")
```

## 📂 `pln/main.py`

Contiene el **bucle principal del bot**.

### 🔄 Flujo por iteración

1. Obtener estado del servidor  
2. Calcular recursos sobrantes y faltantes  
3. Limpiar estados antiguos  
4. Procesar una carta (priorizando confirmaciones)  
5. Decidir si enviar oferta proactiva  
6. Esperar tiempo aleatorio  
7. Repetir  

---

## 🔁 Flujo del sistema

### 1️. Consulta estado

El agente consulta:

```
GET /info
```

Obtiene:

- Recursos actuales  
- Objetivos  
- Alias propio  
- Buzón  

---

### 2️. Análisis estratégico

Se calculan:

- **Recursos sobrantes** → recursos que puede ofrecer  
- **Recuross faltantes** → recursos necesarios para cumplir el objetivo  

---

### 3️. Procesamiento del buzón

Si hay cartas:

- Prioridad a confirmaciones pendientes  

Si no:

- Extraer oferta (OFERTA_V1 → regex → LLM)  
- Aceptar solo si:
  - Lo ofrecido ∈ faltantes  
  - Lo pedido ∈ sobrantes  

---

### 4. Gestión de pendientes

Cuando el agente envía una oferta y el otro jugador confirma:

- Se envía el recurso prometido  
- Se elimina la oferta pendiente  

---

### 5️. Oferta proactiva (probabilística)

Con probabilidad `PROB_ENVIAR_OFERTA`:

- Selecciona destinatario aleatorio  
- Respeta cooldown  
- Envía oferta 1x1  
- Registra pendiente  

---

### 6. Espera aleatoria

Para evitar comportamiento determinista:

```python
sleep(random entre SLEEP_MIN y SLEEP_MAX)
```

---

## 🛠️ Instalación del entorno

### 1. Clonar repositorio

```
git clone https://github.com/maritriv/fdi-pln2608.git
cd fdi-pln2608
```

---

### 2. Instalar dependencias

Instalar `uv` si no está instalado:

```
pip install uv
```

Crear entorno e instalar dependencias:

```
uv sync
```

---

### 3. Configurar LLM (Ollama)

Instalar Ollama y descargar modelo:

```
ollama pull qwen3-vl:8b
```

El bot usa:

```
http://localhost:11434/api/generate
```

---

## ▶️ Ejecución

Desde la raíz del proyecto:

```
uv run fdi-pln-2608-p1
```

Para detener el bot:

```
Ctrl + C
```

El sistema captura la señal y finaliza correctamente.

---

## Equipo de desarrollo

Este proyecto fue desarrollado por los siguientes estudiantes del Grado en Ingeniería de Datos e Inteligencia Artificial (UCM): 
- Carlota Salazar Martín
- Marina Triviño de las Heras
