import time
import random

from pln.logger import log
from pln.config import PROB_ENVIAR_OFERTA, SLEEP_MIN, SLEEP_MAX
from pln.api.client import get_info, borrar_carta
from pln.nlp.normalize import es_mi_alias
from pln.state import OFERTAS_PENDIENTES
from pln.trading.offers import limpiar_ofertas_viejas, limpiar_registro_antiguo
from pln.trading.logic import (
    es_carta_confirmacion_pendiente,
    procesar_confirmacion_pendiente,
    evaluar_y_ejecutar_trueque,
)

from pln.game import (
    recursos_que_me_sobran,
    recursos_que_me_faltan,
    elegir_carta_prioritaria,
    enviar_oferta_proactiva,
)


def main() -> int:
    """
    Bucle principal del bot.

    En cada iteración:
      1) Consulta el estado actual al servidor.
      2) Calcula recursos sobrantes y faltantes.
      3) Limpia estados internos (pendientes / cooldown).
      4) Procesa una carta si existe.
      5) Opcionalmente envía una oferta proactiva.
      6) Duerme un tiempo aleatorio.
    """
    try:
        while True:
            # -------------------------------------------------
            # 1) Obtener estado actual del juego desde el servidor
            # -------------------------------------------------
            recursos, objetivos, mi_alias, buzon = get_info()

            # Si hubo error al contactar con el servidor,
            # esperamos un poco y reintentamos.
            if recursos is None:
                time.sleep(2)
                continue

            # -------------------------------------------------
            # 2) Calcular recursos estratégicos
            # -------------------------------------------------
            # - Recursos sobrantes: lo que puedo intercambiar.
            # - Recursos faltantes: lo que necesito conseguir.
            sobrantes = recursos_que_me_sobran(recursos, objetivos)
            faltantes = recursos_que_me_faltan(recursos, objetivos)

            # Mostrar estado actual por consola
            log("=== ESTADO DEL JUEGO ===")
            log(f"Mi alias: {mi_alias}")
            log(f"Recursos sobrantes: {sobrantes}")
            log(f"Recursos faltantes: {faltantes}")
            log(f"Ofertas pendientes: {OFERTAS_PENDIENTES}")

            # -------------------------------------------------
            # 3) Limpieza de estado interno
            # -------------------------------------------------
            # - Elimina ofertas pendientes demasiado antiguas.
            # - Limpia registros anti-spam viejos.
            limpiar_ofertas_viejas()
            limpiar_registro_antiguo()

            # -------------------------------------------------
            # 4) Procesamiento del buzón
            # -------------------------------------------------
            if buzon:
                # Elegimos primero confirmaciones pendientes si existen
                carta = elegir_carta_prioritaria(
                    buzon, es_carta_confirmacion_pendiente
                )

                # Verificamos que la carta no sea nuestra
                if carta and not es_mi_alias(carta.get("remi"), mi_alias):

                    # Si es confirmación de una oferta pendiente,
                    # intentamos completar el intercambio.
                    if es_carta_confirmacion_pendiente(carta):
                        ok, motivo = procesar_confirmacion_pendiente(
                            carta, recursos, mi_alias, sobrantes
                        )

                    # Si no es confirmación, evaluamos si nos interesa el trueque.
                    else:
                        ok, motivo = evaluar_y_ejecutar_trueque(
                            carta, recursos, mi_alias, sobrantes, faltantes
                        )

                    # Mostramos resultado de la decisión.
                    log(("✅ " if ok else "❌ ") + motivo)

                    # Una vez procesada, eliminamos la carta del buzón.
                    borrar_carta(carta["id"])

                # -------------------------------------------------
                # 5) Alternancia aleatoria: enviar oferta o no
                # -------------------------------------------------
                # Para que el bot no sea determinista,
                # decidimos aleatoriamente si enviar oferta.
                if random.random() < PROB_ENVIAR_OFERTA:
                    enviar_oferta_proactiva(mi_alias, sobrantes, faltantes)

            else:
                # Si el buzón está vacío, intentamos enviar oferta.
                enviar_oferta_proactiva(mi_alias, sobrantes, faltantes)

            # -------------------------------------------------
            # 6) Espera aleatoria antes del siguiente ciclo
            # -------------------------------------------------
            # Esto evita comportamiento demasiado mecánico.
            sleep_s = random.randint(SLEEP_MIN, SLEEP_MAX)
            log(f"Sleep {sleep_s}s")
            print("-" * 40)
            time.sleep(sleep_s)

    except KeyboardInterrupt:
        # Salida limpia cuando el usuario presiona Ctrl+C
        log("👋 Salida solicitada por el usuario (Ctrl+C). Bot detenido correctamente.")
        return 0

    except Exception as e:
        # Cualquier otro error inesperado
        log(f"💥 Error inesperado: {e}")
        raise