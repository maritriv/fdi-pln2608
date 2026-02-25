import time
from pln.state import ULTIMO_ENVIO_A, OFERTAS_PENDIENTES
from pln.config import COOLDOWN_SEGUNDOS

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

def registrar_oferta(dest, ofrezco, pido):
    OFERTAS_PENDIENTES[dest] = {"ofrezco": ofrezco, "pido": pido, "ts": time.time()}

def limpiar_ofertas_viejas(segundos=900):
    ahora = time.time()
    for dest, data in list(OFERTAS_PENDIENTES.items()):
        if ahora - data.get("ts", 0) > segundos:
            del OFERTAS_PENDIENTES[dest]

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