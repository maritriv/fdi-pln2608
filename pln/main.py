import time
import re
import json
import random
import httpx

BASE_URL = "http://147.96.84.134:7719"

# ---------------------------
# Anti-spam (cooldown)
# ---------------------------
ULTIMO_ENVIO_A = {}
COOLDOWN_SEGUNDOS = 180

PAQUETES_ENVIADOS = set()
OFERTAS_PENDIENTES = {}   # dest -> {"ofrezco": str, "pido": str, "ts": float}

PROB_ENVIAR_OFERTA = 0.50
SLEEP_MIN = 5
SLEEP_MAX = 12


# ---------------------------
# Logging simple
# ---------------------------
def log(msg):
    print(f"[BOT] {msg}")


# ---------------------------
# Helpers: seguridad alias
# ---------------------------
def norm_alias(a):
    return (a or "").strip()

def es_mi_alias(alias, mi_alias):
    return norm_alias(alias).lower() == norm_alias(mi_alias).lower()


def extraer_mi_alias_desde_info(data):
    """
    /info puede devolver Alias como:
      - string: "águila intrépido"
      - lista:  ["águila intrépido"]
      - (casos raros) lista vacía / None
    """
    alias_field = data.get("Alias", None)

    if isinstance(alias_field, str):
        a = alias_field.strip()
        return a if a else "Desconocido"

    if isinstance(alias_field, list):
        if not alias_field:
            return "Desconocido"
        # primer alias no vacío
        for x in alias_field:
            if isinstance(x, str) and x.strip():
                return x.strip()
        return "Desconocido"

    # fallback
    return "Desconocido"


# ---------------------------
# Interacción con Ollama
# ---------------------------
def preguntar_llm(prompt):
    try:
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": "qwen3-vl:8b", "prompt": prompt, "stream": False},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    except Exception as e:
        log(f"LLM error: {e}")
        return ""


# ---------------------------
# Recursos
# ---------------------------
def recursos_que_me_sobran(recursos, objetivo):
    sobrantes = {}
    for recurso, tengo in recursos.items():
        necesito = objetivo.get(recurso, 0)
        if tengo > necesito:
            sobrantes[recurso] = tengo - necesito
    return sobrantes


def recursos_que_me_faltan(recursos, objetivo):
    faltantes = {}
    for recurso, necesito in objetivo.items():
        tengo = recursos.get(recurso, 0)
        if tengo < necesito:
            faltantes[recurso] = necesito - tengo
    return faltantes


# ---------------------------
# Info del juego
# ---------------------------
def get_info():
    try:
        response = httpx.get(f"{BASE_URL}/info", timeout=5.0)
        response.raise_for_status()
        data = response.json()

        recursos = data.get("Recursos", {})
        objetivos = data.get("Objetivo", {})
        mi_alias = extraer_mi_alias_desde_info(data)
        buzon = data.get("Buzon", {})

        return recursos, objetivos, mi_alias, buzon
    except Exception as e:
        log(f"Error al obtener info: {e}")
        return None, None, None, None


def get_gente():
    try:
        response = httpx.get(f"{BASE_URL}/gente", timeout=5.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        log(f"Error al obtener gente: {e}")
        return []


# ---------------------------
# Mail + paquetes
# ---------------------------
def borrar_carta(uid):
    try:
        response = httpx.delete(f"{BASE_URL}/mail/{uid}", timeout=5.0)
        if response.status_code == 200:
            log(f"Carta {uid} eliminada del buzón.")
        else:
            log(f"No se pudo eliminar carta {uid}. Status: {response.status_code}")
    except Exception as e:
        log(f"Error al eliminar carta {uid}: {e}")


def enviar_carta(remi, dest, asunto, cuerpo, es_oferta=False):
    if es_mi_alias(dest, remi):
        log(f"(seguridad) Bloqueo: intento de enviarme carta a mí misma ({dest}).")
        return False

    payload = {
        "remi": remi,
        "dest": dest,
        "asunto": asunto,
        "cuerpo": cuerpo,
        "id": "",
        "fecha": ""
    }
    try:
        response = httpx.post(f"{BASE_URL}/carta", json=payload, timeout=5.0)
        response.raise_for_status()
        icono = "✉️  " if es_oferta else ""
        log(f"{icono}Carta enviada a {dest} | Asunto: {asunto}")
        return True
    except Exception as e:
        log(f"Error al enviar carta a {dest}: {e}")
        return False


def enviar_paquete(dest, items, mi_alias=None):
    if mi_alias and es_mi_alias(dest, mi_alias):
        log(f"(seguridad) Bloqueo: intento de enviarme paquete a mí misma ({dest}).")
        return False

    try:
        response = httpx.post(f"{BASE_URL}/paquete/{dest}", json=items, timeout=5.0)
        response.raise_for_status()
        log(f"Paquete enviado a {dest}: {items}")
        return True
    except Exception as e:
        log(f"Error al enviar paquete a {dest}: {e}")
        return False


def enviar_carta_confirmacion(mi_alias, dest, pago_real, premio_real):
    asunto = "Confirmación de trueque"
    cuerpo = (
        f"¡Hecho! Acepto el cambio 🙂\n"
        f"Te he enviado 1 de {pago_real}.\n"
        f"Cuando puedas, envíame 1 de {premio_real}."
    )
    log(f"Enviando confirmación a {dest} (envié {pago_real}, espero {premio_real})")
    enviar_carta(mi_alias, dest, asunto, cuerpo)


def enviar_carta_confirmacion_devolucion(mi_alias, dest, ofrezco):
    asunto = "Confirmación de envío"
    cuerpo = (
        f"¡Gracias! Recibido 🙂\n"
        f"Te envío 1 de {ofrezco} como acordamos."
    )
    log(f"Enviando confirmación a {dest} (devuelvo 1 {ofrezco})")
    enviar_carta(mi_alias, dest, asunto, cuerpo)


# ---------------------------
# Normalización + parsers
# ---------------------------
def normalizar_recurso(nombre):
    if not nombre:
        return ""
    txt = str(nombre).strip().lower()
    txt = re.sub(r"[^a-záéíóúñü]+", "", txt)
    if txt.endswith("s") and len(txt) > 3:
        txt = txt[:-1]
    return txt


def normalizar_texto_libre(s):
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-záéíóúñü0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def es_carta_del_sistema(carta):
    remi = (carta.get("remi", "") or "").strip().lower()
    return remi in {"sistema", "system", "admin"}


def parsear_json_llm(texto):
    if not texto:
        return {}
    texto = texto.strip()

    try:
        return json.loads(texto)
    except Exception:
        pass

    try:
        ini = texto.find("{")
        fin = texto.rfind("}")
        if ini != -1 and fin != -1 and fin > ini:
            return json.loads(texto[ini:fin + 1])
    except Exception:
        return {}

    return {}


def parse_oferta_v1(cuerpo):
    if not cuerpo or "[OFERTA_V1]" not in cuerpo:
        return [], []

    m_quiero = re.search(r"quiero\s*=\s*(\{.*?\})", cuerpo)
    m_ofrezco = re.search(r"ofrezco\s*=\s*(\{.*?\})", cuerpo)

    quiere, ofrece = [], []
    try:
        if m_quiero:
            d = json.loads(m_quiero.group(1))
            quiere = [normalizar_recurso(k) for k in d.keys()]
        if m_ofrezco:
            d = json.loads(m_ofrezco.group(1))
            ofrece = [normalizar_recurso(k) for k in d.keys()]
    except Exception:
        return [], []

    return quiere, ofrece


def extraer_oferta_1x1_regex(texto):
    if not texto:
        return None, None

    t = " ".join(texto.lower().split())
    patrones = [
        r"te\s+doy\s+1\s+(\w+)\s+a\s+cambio\s+de\s+1\s+(\w+)",
        r"ofrezco\s*1\s+(\w+)\s+por\s+1\s+(\w+)",
        r"cambio\s+1\s+(\w+)\s+por\s+1\s+(\w+)",
        r"1\s+(\w+)\s+por\s+1\s+(\w+)",
    ]
    for pat in patrones:
        m = re.search(pat, t)
        if m:
            return normalizar_recurso(m.group(1)), normalizar_recurso(m.group(2))
    return None, None


# ---------------------------
# LLM: interpretación a listas (adaptable)
# ---------------------------
def interpretar_carta_a_listas(carta):
    prompt = f"""
[ROLE: SYSTEM]
Eres un extractor de trueques. NO negocias, NO redactas respuestas.
Solo extraes recursos del texto, aunque esté informal o con listas.

Devuelve SOLO JSON válido (sin texto extra) con:
{{ "quiere": ["..."], "ofrece": ["..."] }}

Reglas:
- No inventes recursos.
- Si hay cantidades, ignóralas (solo nombres).
- "acepto otros recursos" no añade nada si no se listan.
- "tengo X" cuenta como OFRECE solo si sugiere intercambio (por/a cambio/intercambio).

CARTA:
Asunto: {carta.get("asunto")}
Contenido:
{carta.get("cuerpo")}
"""
    raw = preguntar_llm(prompt)
    data = parsear_json_llm(raw)
    quiere = [normalizar_recurso(x) for x in data.get("quiere", []) if normalizar_recurso(x)]
    ofrece = [normalizar_recurso(x) for x in data.get("ofrece", []) if normalizar_recurso(x)]
    return {"quiere": quiere, "ofrece": ofrece}


# ---------------------------
# Confirmaciones de ofertas pendientes (prioridad)
# ---------------------------
def es_carta_confirmacion_pendiente(carta):
    if es_carta_del_sistema(carta):
        return False

    remi = carta.get("remi", "")
    if remi not in OFERTAS_PENDIENTES:
        return False

    data = OFERTAS_PENDIENTES[remi]
    pido = normalizar_recurso(data.get("pido", ""))

    texto = normalizar_texto_libre((carta.get("asunto", "") or "") + " " + (carta.get("cuerpo", "") or ""))

    señales = ["confirm", "envi", "te he enviado", "te envi", "recibid", "hecho", "acepto", "vale", "ok", "de acuerdo"]
    if any(s in texto for s in señales):
        return True

    if pido and pido in texto.replace(" ", ""):
        return True

    return False


def procesar_confirmacion_pendiente(carta, recursos, objetivos, mi_alias, sobrantes, faltantes):
    remi = carta.get("remi", "")
    if remi not in OFERTAS_PENDIENTES:
        return False, "no hay oferta pendiente para este remitente"

    if es_mi_alias(remi, mi_alias):
        return False, "carta enviada por mí misma (ignorada)"

    oferta = OFERTAS_PENDIENTES[remi]
    ofrezco = oferta.get("ofrezco")
    pido = oferta.get("pido")

    if not ofrezco or not pido:
        return False, "oferta pendiente inválida (faltan campos)"

    sobrantes_norm = {normalizar_recurso(k): k for k in sobrantes.keys()}
    ofrezco_norm = normalizar_recurso(ofrezco)

    if ofrezco_norm not in sobrantes_norm:
        return False, f"no envío {ofrezco} porque ya no me sobra (regla estricta)"

    ofrezco_real = sobrantes_norm[ofrezco_norm]
    if recursos.get(ofrezco_real, 0) < 1:
        return False, f"no tengo {ofrezco_real} ahora mismo"

    log(f"(pendiente) Confirmación detectada de {remi}. Envío 1 {ofrezco_real} (yo pedía {pido}).")
    ok = enviar_paquete(remi, {ofrezco_real: 1}, mi_alias=mi_alias)
    if not ok:
        return False, "falló el envío del paquete"

    enviar_carta_confirmacion_devolucion(mi_alias, remi, ofrezco_real)
    del OFERTAS_PENDIENTES[remi]
    return True, f"cerrada oferta pendiente: envié 1 {ofrezco_real}"


# ---------------------------
# Decisión: aceptar oferta y hacer trueque
# ---------------------------
def evaluar_y_ejecutar_trueque(carta, recursos, objetivos, mi_alias, sobrantes, faltantes):
    if es_carta_del_sistema(carta):
        return False, "carta del sistema"

    remi = carta.get("remi", "desconocido")
    if es_mi_alias(remi, mi_alias):
        return False, "carta enviada por mí misma (ignorada)"

    carta_id = carta.get("id", "")
    if carta_id and carta_id in PAQUETES_ENVIADOS:
        return False, "ya procesada (paquete ya enviado)"

    faltantes_norm = {normalizar_recurso(k): k for k in faltantes.keys()}
    sobrantes_norm = {normalizar_recurso(k): k for k in sobrantes.keys()}

    cuerpo = carta.get("cuerpo", "") or ""

    log("Paso 1/4: Interpretando carta (OFERTA_V1 -> regex -> LLM)...")

    quiere, ofrece = parse_oferta_v1(cuerpo)

    if not quiere or not ofrece:
        o, q = extraer_oferta_1x1_regex(cuerpo)
        if o and q:
            ofrece = [o]
            quiere = [q]

    if not quiere or not ofrece:
        data = interpretar_carta_a_listas(carta)
        quiere = data["quiere"]
        ofrece = data["ofrece"]

    log(f"Extraído -> quiere={quiere} ofrece={ofrece}")

    if not quiere or not ofrece:
        return False, "no entiendo la oferta (no pude extraer quiere/ofrece)"

    log("Paso 2/4: ¿Me ofrecen algo que me falta?")
    candidatos_premio = [r for r in ofrece if r in faltantes_norm]
    if not candidatos_premio:
        return False, "no me interesa (no ofrece nada que me falte)"

    log("Paso 3/4: ¿Me piden algo que me sobra?")
    candidatos_pago = [r for r in quiere if r in sobrantes_norm]
    if not candidatos_pago:
        return False, "no puedo pagar (me pide recursos que NO me sobran)"

    pago_real = sobrantes_norm[candidatos_pago[0]]
    premio_real = faltantes_norm[candidatos_premio[0]]

    if recursos.get(pago_real, 0) < 1:
        return False, f"no puedo pagar (no tengo {pago_real})"

    log(f"Paso 4/4: Acepto. Envío 1 {pago_real} a {remi} (espero 1 {premio_real}).")
    ok = enviar_paquete(remi, {pago_real: 1}, mi_alias=mi_alias)
    if not ok:
        return False, "falló el envío del paquete"

    if carta_id:
        PAQUETES_ENVIADOS.add(carta_id)

    enviar_carta_confirmacion(mi_alias, remi, pago_real, premio_real)
    return True, f"envié 1 {pago_real} esperando 1 {premio_real}"


# ---------------------------
# Ofertas proactivas + pendientes
# ---------------------------
def puedo_escribir_a(dest, ahora=None):
    if ahora is None:
        ahora = time.time()
    ultimo = ULTIMO_ENVIO_A.get(dest)
    if ultimo is None:
        return True
    return (ahora - ultimo) >= COOLDOWN_SEGUNDOS


def marcar_envio(dest, ahora=None):
    if ahora is None:
        ahora = time.time()
    ULTIMO_ENVIO_A[dest] = ahora


def limpiar_registro_antiguo(ahora=None):
    if ahora is None:
        ahora = time.time()
    umbral = ahora - (COOLDOWN_SEGUNDOS * 10)
    for dest, ts in list(ULTIMO_ENVIO_A.items()):
        if ts < umbral:
            del ULTIMO_ENVIO_A[dest]


def elegir_intercambio_1x1(sobrantes, faltantes):
    if not sobrantes or not faltantes:
        return None, None
    ofrezco = next(iter(sobrantes.keys()))
    pido = next(iter(faltantes.keys()))
    return ofrezco, pido


def crear_mensaje_oferta_1x1(ofrezco, pido):
    return (
        f"[OFERTA_V1] quiero={{\"{pido}\": 1}} ofrezco={{\"{ofrezco}\": 1}}\n"
        f"Te propongo 1x1: yo te doy 1 {ofrezco} si tú me das 1 {pido}."
    )


def registrar_oferta(dest, ofrezco, pido):
    OFERTAS_PENDIENTES[dest] = {"ofrezco": ofrezco, "pido": pido, "ts": time.time()}


def limpiar_ofertas_viejas(segundos=900):
    ahora = time.time()
    for dest, data in list(OFERTAS_PENDIENTES.items()):
        if ahora - data.get("ts", 0) > segundos:
            del OFERTAS_PENDIENTES[dest]


# ---------------------------
# Selección de carta con prioridad
# ---------------------------
def elegir_carta_prioritaria(buzon):
    cartas = list(buzon.values())
    for c in cartas:
        if es_carta_confirmacion_pendiente(c):
            return c
    return cartas[0] if cartas else None


# ---------------------------
# Main loop (con prioridad confirmaciones)
# ---------------------------
def main():
    while True:
        recursos, objetivos, mi_alias, buzon = get_info()
        if recursos is None:
            time.sleep(2)
            continue

        sobrantes = recursos_que_me_sobran(recursos, objetivos)
        faltantes = recursos_que_me_faltan(recursos, objetivos)

        log("=== ESTADO DEL JUEGO ===")
        log(f"Mi alias: {mi_alias}")
        log(f"Recursos que tengo: {recursos}")
        log(f"Recursos que necesito: {objetivos}")
        log(f"Recursos que me sobran: {sobrantes}")
        log(f"Recursos que me faltan: {faltantes}")
        log(f"Ofertas pendientes: {OFERTAS_PENDIENTES}")

        limpiar_ofertas_viejas()
        limpiar_registro_antiguo()

        if buzon:
            log("Hay cartas en el buzón -> voy a leer una carta (prioridad: confirmaciones).")

            carta = elegir_carta_prioritaria(buzon)
            if not carta:
                log("Buzón raro (vacío).")
            else:
                if es_mi_alias(carta.get("remi"), mi_alias):
                    log("(seguridad) Ignoro carta enviada por mí misma.")
                    borrar_carta(carta["id"])
                else:
                    log(f"LEO CARTA de {carta.get('remi')} | Asunto: {carta.get('asunto')}")
                    log(f"Contenido:\n{carta.get('cuerpo')}")

                    if es_carta_confirmacion_pendiente(carta):
                        log("Detecté carta de CONFIRMACIÓN para una oferta pendiente.")
                        ok, motivo = procesar_confirmacion_pendiente(
                            carta, recursos, objetivos, mi_alias, sobrantes, faltantes
                        )
                        log(("✅ PENDIENTE CERRADA: " if ok else "❌ No pude cerrar pendiente: ") + motivo)
                    else:
                        ok, motivo = evaluar_y_ejecutar_trueque(
                            carta, recursos, objetivos, mi_alias, sobrantes, faltantes
                        )
                        log(("✅ TRUEQUE HECHO: " if ok else "❌ NO HICE TRUEQUE: ") + motivo)

                    log("Borro la carta procesada.")
                    borrar_carta(carta["id"])

            if random.random() < PROB_ENVIAR_OFERTA:
                log("Decisión aleatoria: intentar enviar una oferta proactiva.")
                if sobrantes and faltantes:
                    gente = get_gente()
                    destinatarios = [
                        p.get("alias") for p in gente
                        if p.get("alias") and not es_mi_alias(p.get("alias"), mi_alias)
                    ]

                    if not destinatarios:
                        log("No hay destinatarios válidos (solo estaba yo).")
                    else:
                        dest = random.choice(destinatarios)
                        if puedo_escribir_a(dest):
                            ofrezco, pido = elegir_intercambio_1x1(sobrantes, faltantes)
                            if ofrezco and pido:
                                cuerpo = crear_mensaje_oferta_1x1(ofrezco, pido)
                                log(f"Envío oferta a {dest}: doy {ofrezco} pido {pido}")
                                if enviar_carta(mi_alias, dest, "Propuesta 1x1", cuerpo, es_oferta=True):
                                    marcar_envio(dest)
                                    registrar_oferta(dest, ofrezco, pido)
                        else:
                            log(f"(anti-spam) No escribo a {dest} todavía.")
                else:
                    log("No puedo enviar oferta: no tengo sobrantes o no tengo faltantes.")
            else:
                log("Decisión aleatoria: seguiré leyendo cartas en el próximo ciclo.")

        else:
            log("Buzón vacío.")
            log("Intento enviar una oferta (si puedo).")
            if sobrantes and faltantes:
                gente = get_gente()
                destinatarios = [
                    p.get("alias") for p in gente
                    if p.get("alias") and not es_mi_alias(p.get("alias"), mi_alias)
                ]
                if not destinatarios:
                    log("No envío oferta: no hay destinatarios válidos (solo yo).")
                else:
                    dest = random.choice(destinatarios)
                    if puedo_escribir_a(dest):
                        ofrezco, pido = elegir_intercambio_1x1(sobrantes, faltantes)
                        if ofrezco and pido:
                            cuerpo = crear_mensaje_oferta_1x1(ofrezco, pido)
                            log(f"Envío oferta a {dest}: doy {ofrezco} pido {pido}")
                            if enviar_carta(mi_alias, dest, "Propuesta 1x1", cuerpo, es_oferta=True):
                                marcar_envio(dest)
                                registrar_oferta(dest, ofrezco, pido)
                    else:
                        log(f"(anti-spam) No escribo a {dest} todavía.")
            else:
                log("No envío oferta: no tengo sobrantes o no tengo faltantes.")

        sleep_s = random.randint(SLEEP_MIN, SLEEP_MAX)
        log(f"Sleep {sleep_s}s")
        print("-" * 40)
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()