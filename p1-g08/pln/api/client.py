"""Cliente HTTP para comunicarse con el servidor Butler.

Este módulo aísla todos los accesos a la API del servidor:
consulta de estado, listado de agentes, gestión del buzón,
envío de cartas y envío de paquetes.

Todas las funciones capturan errores de red o de servidor para evitar
que el agente se detenga durante la simulación.
"""

import httpx

from pln.config import BASE_URL, AGENTE
from pln.logger import log
from pln.nlp.normalize import (
    es_mi_alias,
    extraer_mi_alias_desde_info,
)


def get_info() -> tuple[dict | None, dict | None, str | None, dict | None]:
    """Consulta el estado actual del agente en Butler.

    Devuelve una tupla con:
    - recursos actuales,
    - objetivo del agente,
    - alias propio,
    - buzón de cartas.

    Si ocurre un error, devuelve cuatro valores None.
    """
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


def get_gente() -> list:
    """Obtiene la lista de agentes conectados al servidor Butler.

    En modo monopuesto, la petición incluye el parámetro `agente`.
    Si ocurre un error, devuelve una lista vacía.
    """
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


def borrar_carta(uid: str) -> None:
    """Elimina una carta del buzón después de procesarla."""
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


def enviar_carta(
    remi: str,
    dest: str,
    asunto: str,
    cuerpo: str,
    es_oferta: bool = False,
) -> bool:
    """Envía una carta a otro agente.

    Antes de enviarla, comprueba que el destinatario no coincida
    con el propio remitente para evitar autoenvíos accidentales.
    """
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


def enviar_paquete(
    dest: str,
    items: dict,
    mi_alias: str | None = None,
) -> bool:
    """Envía un paquete de recursos a otro agente.

    El paquete representa la acción final de un trueque aceptado.
    En modo monopuesto se incluye el parámetro `agente` para que
    Butler identifique correctamente al agente emisor.
    """
    if mi_alias and es_mi_alias(dest, mi_alias):
        log(f"(seguridad) Bloqueo: intento de enviarme paquete a mí misma ({dest}).")
        return False

    try:
        r = httpx.post(
            f"{BASE_URL}/paquete/{dest}",
            params={"agente": AGENTE} if AGENTE else None,
            json=items,
            timeout=5.0,
        )
        r.raise_for_status()

        log(f"Paquete enviado a {dest}: {items}")

        return True

    except Exception as e:
        log(f"Error al enviar paquete a {dest}: {e}")
        return False
