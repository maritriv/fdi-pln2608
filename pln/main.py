import json
import requests

BUTLER_BASE = "http://147.96.81.252:8000"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3-vl:8b"

ALIAS = "fdi-pln-2608"

RECURSOS_CLASE = {
    "oro": 4,
    "piedra": 4,
    "madera": 1,
    "trigo": 3,
}


def ollama_json(prompt: str) -> dict:
    payload = {
	"model": MODEL,
	"messages": [{"role": "user", "content": prompt}],
	"stream": False,
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()

    text = r.json()["message"]["content"].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Ollama no devolvió JSON válido:\n{text}")

    return json.loads(text[start : end + 1])


def enviar_paquete(recursos: dict):
    url = f"{BUTLER_BASE}/paquete"
    params = {"dest": ALIAS} # 🔴 QUERY PARAM
    r = requests.post(url, params=params, json=recursos, timeout=30)
    return r.status_code, r.text


def leer_info():
    r = requests.get(f"{BUTLER_BASE}/info", timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    info = leer_info()

    prompt = (
	"Teniendo en cuenta los recursos disponibles:\n"
	f"{info['Recursos']}\n"
	"y el objetivo:\n"
	f"{info['Objetivo']}\n"
	"decide un paquete de recursos VALIDO.\n"
	"Devuelve SOLO un JSON con claves oro, piedra, madera, trigo.\n"
	"No pidas más recursos de los disponibles."
    )
    recursos = ollama_json(prompt)
    print("Recursos generados por Ollama:", recursos)

    code, body = enviar_paquete(recursos)
    print("Respuesta Butler /paquete:", code, body)

    info = leer_info()
    print("Recursos actuales en Butler:", info.get("Recursos"))


if __name__ == "__main__":
    main()
