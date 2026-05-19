"""Gestión de ofertas proactivas y control anti-spam.

Este módulo mantiene el registro de ofertas enviadas, controla el cooldown
entre mensajes al mismo agente y construye mensajes de oferta 1x1.
"""

import time

from pln.config import COOLDOWN_SEGUNDOS
from pln.state import OFERTAS_PENDIENTES, ULTIMO_ENVIO_A


def puedo_escribir_a(dest: str, ahora: float | None = None) -> bool:
    """Indica si podemos escribir a un destinatario respetando el cooldown."""
    if ahora is None:
        ahora = time.time()

    ultimo = ULTIMO_ENVIO_A.get(dest)

    if ultimo is None:
        return True

    return (ahora - ultimo) >= COOLDOWN_SEGUNDOS


def marcar_envio(dest: str, ahora: float | None = None) -> None:
    """Registra el momento en el que se envió una carta a un destinatario."""
    if ahora is None:
        ahora = time.time()

    ULTIMO_ENVIO_A[dest] = ahora


def limpiar_registro_antiguo(ahora: float | None = None) -> None:
    """Elimina entradas antiguas del registro anti-spam."""
    if ahora is None:
        ahora = time.time()

    umbral = ahora - (COOLDOWN_SEGUNDOS * 10)

    for dest, ts in list(ULTIMO_ENVIO_A.items()):
        if ts < umbral:
            del ULTIMO_ENVIO_A[dest]


def registrar_oferta(dest: str, ofrezco: str, pido: str) -> None:
    """Guarda una oferta enviada como pendiente de confirmación."""
    OFERTAS_PENDIENTES[dest] = {
        "ofrezco": ofrezco,
        "pido": pido,
        "ts": time.time(),
    }


def limpiar_ofertas_viejas(segundos: int = 900) -> None:
    """Elimina ofertas pendientes que llevan demasiado tiempo abiertas."""
    ahora = time.time()

    for dest, data in list(OFERTAS_PENDIENTES.items()):
        if ahora - data.get("ts", 0) > segundos:
            del OFERTAS_PENDIENTES[dest]


def elegir_intercambio_1x1(
    sobrantes: dict,
    faltantes: dict,
) -> tuple[str | None, str | None]:
    """Elige un intercambio simple 1x1 usando el primer sobrante y faltante."""
    if not sobrantes or not faltantes:
        return None, None

    ofrezco = next(iter(sobrantes.keys()))
    pido = next(iter(faltantes.keys()))

    return ofrezco, pido


def crear_mensaje_oferta_1x1(ofrezco: str, pido: str) -> str:
    """Construye el mensaje textual de una oferta 1x1.

    Incluye una parte estructurada `[OFERTA_V1]` para que otros agentes puedan
    parsearla fácilmente y una frase en lenguaje natural para legibilidad.
    """
    return (
        f'[OFERTA_V1] quiero={{"{pido}": 1}} ofrezco={{"{ofrezco}": 1}}\n'
        f"Te propongo 1x1: yo te doy 1 {ofrezco} si tú me das 1 {pido}."
    )