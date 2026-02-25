from pln.logger import log
from pln.state import PAQUETES_ENVIADOS, OFERTAS_PENDIENTES
from pln.nlp.normalize import (
    es_carta_del_sistema,
    es_mi_alias,
    normalizar_recurso,
    normalizar_texto_libre,
)
from pln.nlp.parse import parse_oferta_v1, extraer_oferta_1x1_regex
from pln.nlp.llm import interpretar_carta_a_listas
from pln.api.client import enviar_paquete, enviar_carta


def enviar_carta_confirmacion(mi_alias, dest, pago_real, premio_real):
    """
    Envía una carta confirmando que hemos aceptado un trueque.

    Se usa cuando:
      - Hemos decidido aceptar una oferta recibida.
      - Ya hemos enviado el recurso solicitado.
      - Esperamos que el otro jugador nos envíe su parte.

    Parámetros:
        mi_alias (str): Nuestro alias.
        dest (str): Destinatario.
        pago_real (str): Recurso que enviamos.
        premio_real (str): Recurso que esperamos recibir.
    """
    asunto = "Confirmación de trueque"
    cuerpo = (
        f"¡Hecho! Acepto el cambio 🙂\n"
        f"Te he enviado 1 de {pago_real}.\n"
        f"Cuando puedas, envíame 1 de {premio_real}."
    )

    log(f"Enviando confirmación a {dest} (envié {pago_real}, espero {premio_real})")
    enviar_carta(mi_alias, dest, asunto, cuerpo)


def enviar_carta_confirmacion_devolucion(mi_alias, dest, ofrezco):
    """
    Envía una carta cuando estamos completando una oferta pendiente.

    Se usa cuando:
      - Nosotros enviamos una oferta.
      - El otro jugador la acepta.
      - Ahora enviamos el recurso prometido.

    Parámetros:
        mi_alias (str): Nuestro alias.
        dest (str): Destinatario.
        ofrezco (str): Recurso que estamos devolviendo.
    """
    asunto = "Confirmación de envío"
    cuerpo = (
        f"¡Gracias! Recibido 🙂\n"
        f"Te envío 1 de {ofrezco} como acordamos."
    )

    log(f"Enviando confirmación a {dest} (devuelvo 1 {ofrezco})")
    enviar_carta(mi_alias, dest, asunto, cuerpo)


def es_carta_confirmacion_pendiente(carta):
    """
    Determina si una carta corresponde a la confirmación
    de una oferta que nosotros enviamos anteriormente.

    Criterios:
      - No debe ser una carta del sistema.
      - El remitente debe tener una oferta pendiente registrada.
      - El texto debe contener señales típicas de confirmación
        o mencionar el recurso que pedíamos.

    Devuelve:
        True si es confirmación pendiente, False en caso contrario.
    """
    # Ignoramos cartas del sistema
    if es_carta_del_sistema(carta):
        return False

    remi = carta.get("remi", "")

    # Solo tiene sentido si tenemos una oferta pendiente con ese remitente
    if remi not in OFERTAS_PENDIENTES:
        return False

    data = OFERTAS_PENDIENTES[remi]
    pido = normalizar_recurso(data.get("pido", ""))

    # Normalizamos texto completo para buscar señales
    texto = normalizar_texto_libre(
        (carta.get("asunto", "") or "") + " " + (carta.get("cuerpo", "") or "")
    )

    # Palabras típicas que indican aceptación o confirmación
    señales = [
        "confirm",
        "envi",
        "te he enviado",
        "te envi",
        "recibid",
        "hecho",
        "acepto",
        "vale",
        "ok",
        "de acuerdo",
    ]

    # Si detectamos alguna señal clara
    if any(s in texto for s in señales):
        return True

    # O si menciona explícitamente el recurso que pedíamos
    if pido and pido in texto.replace(" ", ""):
        return True

    return False


def procesar_confirmacion_pendiente(carta, recursos, mi_alias, sobrantes):
    """
    Procesa una carta que confirma una oferta pendiente.

    Flujo:
      1) Verifica que exista una oferta pendiente.
      2) Comprueba que el recurso prometido aún nos sobra.
      3) Envía el paquete correspondiente.
      4) Envía carta de confirmación.
      5) Elimina la oferta pendiente del estado interno.

    Devuelve:
        (bool, str) -> éxito y mensaje explicativo.
    """
    remi = carta.get("remi", "")

    if remi not in OFERTAS_PENDIENTES:
        return False, "no hay oferta pendiente"

    # Seguridad: no procesamos nuestras cartas
    if es_mi_alias(remi, mi_alias):
        return False, "carta enviada por mí mismo (ignorada)"

    oferta = OFERTAS_PENDIENTES[remi]
    ofrezco = oferta.get("ofrezco")
    pido = oferta.get("pido")

    if not ofrezco or not pido:
        return False, "oferta pendiente inválida"

    # Normalizamos sobrantes para comparar correctamente
    sobrantes_norm = {normalizar_recurso(k): k for k in sobrantes.keys()}
    ofrezco_norm = normalizar_recurso(ofrezco)

    # Solo enviamos si sigue siendo sobrante
    if ofrezco_norm not in sobrantes_norm:
        return False, f"no envío {ofrezco} porque ya no me sobra"

    ofrezco_real = sobrantes_norm[ofrezco_norm]

    if recursos.get(ofrezco_real, 0) < 1:
        return False, f"no tengo {ofrezco_real} ahora"

    log(f"(pendiente) Confirmación de {remi}. Envío 1 {ofrezco_real} (yo pedía {pido}).")

    ok = enviar_paquete(remi, {ofrezco_real: 1}, mi_alias=mi_alias)
    if not ok:
        return False, "falló el envío del paquete"

    enviar_carta_confirmacion_devolucion(mi_alias, remi, ofrezco_real)

    # Cerramos la oferta pendiente
    del OFERTAS_PENDIENTES[remi]

    return True, f"cerrada pendiente: envié 1 {ofrezco_real}"


def evaluar_y_ejecutar_trueque(carta, recursos, mi_alias, sobrantes, faltantes):
    """
    Evalúa una oferta recibida y decide si ejecutarla.

    Flujo:
      1) Ignora cartas del sistema o propias.
      2) Extrae recursos ofrecidos y solicitados.
      3) Comprueba si nos interesa (ofrece algo que nos falta).
      4) Comprueba si podemos pagar (pide algo que nos sobra).
      5) Ejecuta el envío del paquete.
      6) Envía carta de confirmación.

    Devuelve:
        (bool, str) -> éxito y motivo.
    """

    if es_carta_del_sistema(carta):
        return False, "carta del sistema"

    remi = carta.get("remi", "desconocido")

    if es_mi_alias(remi, mi_alias):
        return False, "carta enviada por mí misma (ignorada)"

    carta_id = carta.get("id", "")

    # Evita ejecutar dos veces la misma carta
    if carta_id and carta_id in PAQUETES_ENVIADOS:
        return False, "ya procesada"

    # Normalización de recursos para comparación robusta
    faltantes_norm = {normalizar_recurso(k): k for k in faltantes.keys()}
    sobrantes_norm = {normalizar_recurso(k): k for k in sobrantes.keys()}

    cuerpo = carta.get("cuerpo", "") or ""

    log("Paso 1/4: Interpretando carta (OFERTA_V1 -> regex -> LLM)...")

    # 1️⃣ Intento formato estructurado
    quiere, ofrece = parse_oferta_v1(cuerpo)

    # 2️⃣ Intento regex simple
    if not quiere or not ofrece:
        o, q = extraer_oferta_1x1_regex(cuerpo)
        if o and q:
            ofrece = [o]
            quiere = [q]

    # 3️⃣ Fallback: LLM
    if not quiere or not ofrece:
        data = interpretar_carta_a_listas(carta)
        quiere = data["quiere"]
        ofrece = data["ofrece"]

    log(f"Extraído -> quiere={quiere} ofrece={ofrece}")

    if not quiere or not ofrece:
        return False, "no entiendo la oferta"

    log("Paso 2/4: ¿Me ofrecen algo que me falta?")
    candidatos_premio = [r for r in ofrece if r in faltantes_norm]

    if not candidatos_premio:
        return False, "no me interesa"

    log("Paso 3/4: ¿Me piden algo que me sobra?")
    candidatos_pago = [r for r in quiere if r in sobrantes_norm]

    if not candidatos_pago:
        return False, "no puedo pagar"

    pago_real = sobrantes_norm[candidatos_pago[0]]
    premio_real = faltantes_norm[candidatos_premio[0]]

    if recursos.get(pago_real, 0) < 1:
        return False, f"no tengo {pago_real}"

    log(f"Paso 4/4: Acepto. Envío 1 {pago_real} a {remi} (espero 1 {premio_real}).")

    ok = enviar_paquete(remi, {pago_real: 1}, mi_alias=mi_alias)
    if not ok:
        return False, "falló el envío"

    if carta_id:
        PAQUETES_ENVIADOS.add(carta_id)

    enviar_carta_confirmacion(mi_alias, remi, pago_real, premio_real)

    return True, f"envié 1 {pago_real} esperando 1 {premio_real}"