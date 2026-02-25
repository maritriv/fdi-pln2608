# pln/game.py

import random
from pln.logger import log
from pln.api.client import get_gente, enviar_carta
from pln.nlp.normalize import es_mi_alias
from pln.trading.offers import (
    puedo_escribir_a,
    marcar_envio,
    registrar_oferta,
    elegir_intercambio_1x1,
    crear_mensaje_oferta_1x1,
)


def recursos_que_me_sobran(recursos: dict, objetivo: dict) -> dict:
    """
    Devuelve los recursos que superan el objetivo.
    """
    return {
        r: tengo - objetivo.get(r, 0)
        for r, tengo in recursos.items()
        if tengo > objetivo.get(r, 0)
    }


def recursos_que_me_faltan(recursos: dict, objetivo: dict) -> dict:
    """
    Devuelve los recursos que faltan para alcanzar el objetivo.
    """
    return {
        r: objetivo[r] - recursos.get(r, 0)
        for r in objetivo
        if recursos.get(r, 0) < objetivo[r]
    }


def elegir_carta_prioritaria(buzon: dict, es_confirmacion_fn):
    """
    Devuelve primero una carta que sea confirmación pendiente.
    Si no hay, devuelve la primera disponible.
    """
    cartas = list(buzon.values())
    for c in cartas:
        if es_confirmacion_fn(c):
            return c
    return cartas[0] if cartas else None


def enviar_oferta_proactiva(mi_alias: str, sobrantes: dict, faltantes: dict):
    """
    Envía una oferta 1x1 aleatoria respetando reglas anti-spam.
    """
    if not (sobrantes and faltantes):
        log("No puedo enviar oferta: no tengo sobrantes o no tengo faltantes.")
        return

    gente = get_gente()
    destinatarios = [
        p.get("alias")
        for p in gente
        if p.get("alias") and not es_mi_alias(p.get("alias"), mi_alias)
    ]

    if not destinatarios:
        log("No hay destinatarios válidos porque no hay nadie más conectado al servidor.")
        return

    dest = random.choice(destinatarios)

    if not puedo_escribir_a(dest):
        log(f"(anti-spam) No escribo a {dest} todavía.")
        return

    ofrezco, pido = elegir_intercambio_1x1(sobrantes, faltantes)
    if not ofrezco or not pido:
        log("No pude elegir intercambio 1x1.")
        return

    cuerpo = crear_mensaje_oferta_1x1(ofrezco, pido)

    log(f"Envío oferta a {dest}: doy {ofrezco} pido {pido}")

    if enviar_carta(mi_alias, dest, "Propuesta 1x1", cuerpo, es_oferta=True):
        marcar_envio(dest)
        registrar_oferta(dest, ofrezco, pido)