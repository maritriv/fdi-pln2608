import os

# Para ejeuctar en local
BASE_URL = os.getenv("FDI_PLN__BUTLER_ADDRESS", "http://127.0.0.1:7719").rstrip("/")
AGENTE = os.getenv("FDI_PLN__AGENTE", "tigre")

# Para ejecutar en clase
# BASE_URL = "http://147.96.84.134:7719"

COOLDOWN_SEGUNDOS = 180

PROB_ENVIAR_OFERTA = 0.50
SLEEP_MIN = 5
SLEEP_MAX = 12

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3-vl:8b"
OLLAMA_TIMEOUT = 30
