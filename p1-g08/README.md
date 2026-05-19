# Butler Trading Agent — PLN (Grupo 08)

--------------
## Descripción

Este repositorio contiene la implementación de un **agente autónomo de negociación e intercambio de recursos** desarrollado para la práctica _"Los agentes de Butler"_ en la asignatura de Procesamiento del Lenguaje Natural (Grado en Ingeniería de Datos e Inteligencia Artificial, UCM).

El agente está diseñado para cohabitar en un entorno multiagente competitivo y dinámico. Su objetivo principal es interactuar con otros agentes mediante el envío de cartas (negociación) y paquetes (bienes) a través de un servidor central (Butler) para maximizar el progreso hacia sus propios objetivos de inventario.


-------------------------------------------
## Filosofía de Diseño y Estrategia Híbrida

A diferencia de los enfoques puramente deterministas o completamente basados en LLMs, este proyecto destaca por una arquitectura híbrida de tres niveles. El modelo de lenguaje local (Qwen2.5-VL / Qwen3-VL de 8B vía Ollama) no toma decisiones estratégicas directas; en su lugar, actúa como un extractor semántico avanzado.

La lógica de negocio y la toma de decisiones financieras permanecen blindadas bajo reglas deterministas en código Python. Cuando llega una carta, el agente activa un pipeline de procesamiento lingüístico en cascada:

📥 Carta Recibida
       │
       ▼
┌─────────────────────────────┐
│ 1. Parsing Estructurado    │ ──(¿Formato OFERTA_V1?)──► [Aceptado/Procesado]
└─────────────────────────────┘
       │ No
       ▼
┌─────────────────────────────┐
│ 2. Extracción Regex         │ ──(¿Patrón 1x1 match?)──► [Aceptado/Procesado]
└─────────────────────────────┘
       │ No
       ▼
┌─────────────────────────────┐
│ 3. Inferencia Semántica LLM │ ──(Ollama Inferencia)───► Extract: {quiere, ofrece}
└─────────────────────────────┘

Esta estrategia garantiza cuatro ventajas críticas en el sistema:

- **Robustez Extrema**: Tolerancia absoluta a mensajes con lenguaje natural ambiguo, informal o mal formateado.

- **Eficiencia Computacional**: Si un agente competidor envía un formato estructurado, el pipeline lo procesa instantáneamente mediante Regex, evitando el coste temporal de una llamada al LLM.

- **Seguridad Operacional**: Al delegar la lógica de aceptación ("¿me conviene este trato?") a funciones deterministas basadas en matrices de inventario, eliminamos por completo el riesgo de alucinación del modelo.


-----------------------------------
## Arquitectura Modular del Sistema
El proyecto se rige por el principio de separación de responsabilidades, dividiendo el sistema en módulos independientes localizados dentro del paquete `pln/`:
```
pln/
├── api/          # Capa de transporte y red. Gestiona las peticiones HTTP (GET/POST) 
│   └── client.py # contra los endpoints del servidor Butler (/info, /carta, /paquete).
├── nlp/          # Core de procesamiento lingüístico. Contiene las funciones de sanitización,
│   ├── llm.py    # normalización sintáctica, reglas regex y la interfaz con la API de Ollama.
│   ├── normalize.py
│   └── parse.py
├── trading/      # Motor financiero y de negociación. Valida la viabilidad de los trueques,
│   ├── logic.py  # controla los duplicados de transacciones y gestiona las ofertas activas.
│   └── offers.py
├── state.py      # Gestor del estado interno transaccional y persistencia en memoria viva.
├── game.py       # Cerebro estratégico. Calcula excedentes, faltantes y prioriza acciones.
├── config.py     # Archivo centralizado de hiperparámetros (cooldowns, sleeps, probabilidades).
├── logger.py     # Motor de trazabilidad por consola del comportamiento del agente.
└── main.py       # Punto de entrada. Ejecuta el Agentic Loop infinito.
```


---------------------------------------------
## El Ciclo de Vida del Agente (Agentic Loop)
El agente opera de manera asíncrona y autónoma mediante un bucle continuo (`while True`) estructurado bajo el paradigma Diseñar ➔ Actuar ➔ Observar:


### 1. Sincronización de Estado y Análisis Estratégico

Al inicio de cada iteración, el agente consulta el endpoint `/info` para sincronizar su alias, objetivos y recursos actuales. Inmediatamente después, el módulo `game.py` evalúa de forma dinámica la matriz de necesidades, dividiendo el inventario en dos listas vivas: `recursos_sobrantes` (monedas de cambio) y `recursos_faltantes` (objetivos prioritarios).

### 2. Procesamiento Inteligente del Buzón

El agente lee las cartas entrantes filtrando mensajes del sistema o autoenvíos. Procesa los textos mediante el pipeline de tres niveles (Estructurado ➔ Regex ➔ LLM) para traducir el lenguaje natural a un esquema estricto:
```
{
  "quiere": ["Madera"],
  "ofrece": ["Oro"]
}
```
Una oferta se considera viable **únicamente** si el recurso ofrecido por el rival se encuentra en nuestra lista de `faltantes` y el recurso solicitado pertenece a nuestros `sobrantes`. Para fomentar la liquidez en fases tempranas, el agente cuenta con un margen de generosidad configurable que flexibiliza el valor de los intercambios.

### 3. Resolución Transaccional y Ofertas Proactivas

- **Respuestas y Envío de Paquetes**: Si una oferta es aprobada (o si se detecta una confirmación de un intercambio que nosotros propusimos previamente), el agente ejecuta el envío físico de los recursos mediante `/enviar_paquete` y limpia de forma segura el buzón con `/borrar_carta` para evitar el desbordamiento del contexto.

- **Estrategia Proactiva**: Si no hay mensajes entrantes, el agente calcula mediante una distribución probabilística (`PROB_ENVIAR_OFERTA`) si debe iniciar una negociación. Elige un destinatario del ecosistema de manera aleatoria, comprueba las restricciones de _cooldown anti-spam_ en `offers.py`, genera una propuesta formal y la envía al mercado, registrándola internamente como oferta pendiente.

### 4. Mitigación de Comportamientos Predecibles

Para evitar colisiones de red con otros bots y mitigar patrones de comportamiento que puedan ser explotados por agentes rivales, el bucle finaliza aplicando un retardo estocástico calculado dinámicamente entre un umbral configurable (`SLEEP_MIN` y `SLEEP_MAX`).


----------------------------------------------------
## 🔒 Mecanismos de Robustez y Seguridad Operacional

Para garantizar la estabilidad del agente durante ejecuciones prolongadas en entornos hostiles, se han implementado las siguientes protecciones:

- **Filtro Anti-Spam**: Sistema de cooldown temporal que bloquea envíos masivos hacia un mismo agente si este no responde a ofertas previas.

- **Control de Duplicados**: Validación de hashes de cartas procesadas para evitar la doble ejecución de transacciones idénticas.

- **Tolerancia a Fallos de Red**: Captura de excepciones en la capa de comunicación HTTP con reintentos exponenciales automáticos ante caídas intermitentes del servidor Butler.

- **Cierre Seguro (Graceful Shutdown)**: El script captura de forma nativa la señal SIGINT (Ctrl + C), asegurando que el agente complete cualquier transacción en curso antes de liberar los recursos y desconectarse del ecosistema.


----------------------------------------------
## 🛠️ Instalación y Configuración del Entorno

### Prerrequisitos

El proyecto utiliza uv, un instalador y gestor de entornos de Python extremadamente rápido escrito en Rust.

#### 1. Clonar el repositorio:
```
git clone https://github.com/maritriv/fdi-pln2608.git
cd fdi-pln2608/p1-g08
```

#### 2. Instalar dependencias y sincronizar el entorno virtual:
```
pip install uv
uv sync
```

#### 3. Configurar el entorno local de Ollama:
Asegúrate de que el servicio de Ollama esté ejecutándose localmente (`http://localhost:11434`) y descarga el modelo visual e idiomático utilizado por el pipeline de extracción:
```
ollama pull qwen3-vl:8b
```

---------------------------
## 🚀 Ejecución del Agente

El comportamiento del agente se puede parametrizar mediante variables de entorno antes del lanzamiento.

|Variable|Descripción|Ejemplo de valor|
|--------|-----------|----------------|
|`FDI_PLN__BUTLER_ADDRESS`|URL base del servidor Butler central|`[http://127.0.0.1:7719](http://127.0.0.1:7719)`|
|`FDI_PLN__AGENTE`|Alias único asignado al agente en el servidor|`Mercader_G08`|

Para arrancar el ciclo autónomo del agente, exporta la dirección del servidor y ejecuta el módulo principal con `uv`:
```
export FDI_PLN__BUTLER_ADDRESS="http://127.0.0.1:7719"
uv run fdi-pln-2608-p1
```


--------------------------
## 👥 Equipo de Desarrollo

Proyecto diseñado e implementado por estudiantes del Grado en Ingeniería de Datos e Inteligencia Artificial de la Universidad Complutense de Madrid (UCM):

- **Carlota Salazar Martín**
- **Marina Triviño de las Heras**

