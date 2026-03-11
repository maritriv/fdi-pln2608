import httpx
from pln.config import BASE_URL, AGENTE
from pln.logger import log
from pln.nlp.normalize import es_mi_alias, extraer_mi_alias_desde_info


def get_info():
    try:
        r = httpx.get(
            f"{BASE_URL}/info",
            params={"agente": AGENTE} if AGENTE else None,
            timeout=5.0,
        )
        r.raise_for_status()
        data = r.json()

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
        r = httpx.get(
            f"{BASE_URL}/gente",
            params={"agente": AGENTE} if AGENTE else None,
            timeout=5.0,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f"Error al obtener gente: {e}")
        return []


def borrar_carta(uid):
    try:
        r = httpx.delete(
            f"{BASE_URL}/mail/{uid}",
            params={"agente": AGENTE} if AGENTE else None,
            timeout=5.0,
        )
        if r.status_code == 200:
            log(f"Carta {uid} eliminada del buzón.")
        else:
            log(f"No se pudo eliminar carta {uid}. Status: {r.status_code}")
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
        "fecha": "",
    }
    try:
        r = httpx.post(
            f"{BASE_URL}/carta",
            params={"agente": AGENTE} if AGENTE else None,
            json=payload,
            timeout=5.0,
        )
        r.raise_for_status()
        icono = "✉️ " if es_oferta else ""
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
        r = httpx.post(f"{BASE_URL}/paquete/{dest}", json=items, timeout=5.0)
        r.raise_for_status()
        log(f"Paquete enviado a {dest}: {items}")
        return True
    except Exception as e:
        log(f"Error al enviar paquete a {dest}: {e}")
        return False
