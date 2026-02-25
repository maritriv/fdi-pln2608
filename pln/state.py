"""
Estructuras de estado global del bot.
Mantienen memoria entre iteraciones del bucle principal.
"""

# Registro de último envío por destinatario.
# Clave: alias del jugador
# Valor: timestamp (time.time()) del último mensaje enviado
# Se usa para aplicar cooldown y evitar spam.
ULTIMO_ENVIO_A = {}

# Conjunto de IDs de cartas ya procesadas para las que hemos enviado paquete.
# Evita ejecutar el mismo trueque dos veces.
PAQUETES_ENVIADOS = set()

# Ofertas que hemos enviado y están pendientes de confirmación.
# Clave: alias del destinatario
# Valor: diccionario con:
#   - "ofrezco": recurso que prometimos enviar
#   - "pido": recurso que esperamos recibir
#   - "ts": timestamp en el que se envió la oferta
# Se utiliza para completar el intercambio cuando llegue la carta de confirmación.
OFERTAS_PENDIENTES = {}