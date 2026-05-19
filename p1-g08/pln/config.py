"""Configuración global del agente.

Este módulo centraliza los parámetros de ejecución del bot:
dirección del servidor Butler, alias del agente, tiempos de espera,
probabilidad de envío de ofertas y configuración opcional de Ollama.
"""

import os

# URL del servidor Butler.
# En la entrega debe venir de la variable FDI_PLN__BUTLER_ADDRESS.
# El valor por defecto permite probar en local si el servidor está levantado ahí.
BASE_URL = os.getenv("FDI_PLN__BUTLER_ADDRESS", "http://127.0.0.1:7719").rstrip("/")

# Alias del agente en modo monopuesto.
# Si no existe, se omite el parámetro agente en las peticiones.
AGENTE = os.getenv("FDI_PLN__AGENTE")

# Tiempo mínimo entre ofertas al mismo agente.
COOLDOWN_SEGUNDOS = 180

# Control del comportamiento proactivo del agente.
PROB_ENVIAR_OFERTA = 0.50
SLEEP_MIN = 5
SLEEP_MAX = 12

# Configuración opcional para uso de LLM local con Ollama.
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3-vl:8b"
OLLAMA_TIMEOUT = 30