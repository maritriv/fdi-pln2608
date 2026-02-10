import time
import httpx

BASE_URL = "http://147.96.81.252:7719"
#qwen3-vl:8b

# ---------------------------
# Anti-spam (cooldown)
# ---------------------------
ULTIMO_ENVIO_A = {}  # alias_destino -> timestamp del último envío
COOLDOWN_SEGUNDOS = 180  # 3 minutos (ajusta a tu gusto)
PAQUETES_ENVIADOS = set()  # guardaremos ids de cartas ya procesadas para envío



# ---------------------------
# Interacción con Ollama
# ---------------------------
def preguntar_llm(prompt):
    """
    Envía un prompt a Ollama y devuelve el texto de respuesta
    """
    try:
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen3-vl:8b",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    except Exception as e:
        print("Error al hablar con el LLM:", e)
        return ""



# ---------------------------
# Funciones de recursos
# ---------------------------
def recursos_que_me_sobran(recursos, objetivo):
    sobrantes = {}
    for recurso, cantidad_tengo in recursos.items():
        cantidad_necesaria = objetivo.get(recurso, 0)
        if cantidad_tengo > cantidad_necesaria:
            sobrantes[recurso] = cantidad_tengo - cantidad_necesaria
    return sobrantes


def recursos_que_me_faltan(recursos, objetivo):
    faltantes = {}
    for recurso, cantidad_necesaria in objetivo.items():
        cantidad_tengo = recursos.get(recurso, 0)
        if cantidad_tengo < cantidad_necesaria:
            faltantes[recurso] = cantidad_necesaria - cantidad_tengo
    return faltantes



# ---------------------------
# Funciones de info
# ---------------------------
def get_info():
    try:
        response = httpx.get(f"{BASE_URL}/info", timeout=5.0)
        response.raise_for_status()
        data = response.json()

        recursos = data.get("Recursos", {})
        objetivos = data.get("Objetivo", {})
        aliases = data.get("Alias", ["Desconocido"])  # devuelve una lista de aliases
        mi_alias = aliases[0] # coge el primer elemento de la lista de aliases 
        #alias_mio = aliases[0]

        buzon = data.get("Buzon", {})

        return recursos, objetivos, mi_alias, buzon

    except Exception as e:
        print("Error al obtener info:", e)
        return None, None, None, None


def get_gente():
    try:
        response = httpx.get(f"{BASE_URL}/gente", timeout=5.0)
        response.raise_for_status()
        return response.json()  # lista de aliases
    except Exception as e:
        print("Error al obtener gente:", e)
        return []
    


# ---------------------------
# Evaluar cartas
# ---------------------------
def cartas_utiles(buzon, faltantes):
    utiles = []
    for carta_id, carta in buzon.items():
        cuerpo = carta.get("cuerpo", "").lower()
        if carta_parece_aceptacion(carta):
            utiles.append(carta)
            continue

        for recurso in faltantes:
            if recurso.lower() in cuerpo:
                utiles.append(carta)
                break
    return utiles



def borrar_carta(uid):
    """
    Elimina una carta usando DELETE /mail/{uid}
    """
    try:
        response = httpx.delete(f"{BASE_URL}/mail/{uid}", timeout=5.0)
        if response.status_code == 200:
            print(f"Carta {uid} eliminada del buzón.")
        else:
            print(f"No se pudo eliminar carta {uid}. Status: {response.status_code}")
    except Exception as e:
        print(f"Error al eliminar carta {uid}:", e)


def enviar_carta(remi, dest, asunto, cuerpo):
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
        print(f"Carta enviada a {dest} correctamente.")
    except Exception as e:
        print(f"Error al enviar carta a {dest}:", e)


def interpretar_carta(carta, faltantes):
    prompt = f"""
        [ROLE: SYSTEM]
        Eres un analizador estricto de mensajes para un juego de trueque.
        NO negocias, NO propones intercambios, NO redactas respuestas.
        Tu única función es EXTRAER información explícita y evaluarla.

        REGLAS ABSOLUTAS (PRIORIDAD MÁXIMA):
        - NO inventes recursos, intenciones ni condiciones.
        - NO infieras nada que no esté claramente escrito.
        - Si algo no está explícito, escribe exactamente: "no claro".
        - NO añadas texto fuera del formato pedido.

        CONTEXTO DEL JUEGO:
        - Los jugadores intercambian recursos.
        - El objetivo es conseguir los recursos necesarios lo antes posible.
        - El interés se mide SOLO según utilidad y claridad del mensaje recibido.

        CARTA RECIBIDA:
        Remitente: {carta['remi']}
        Asunto: {carta['asunto']}
        Contenido:
        {carta['cuerpo']}

        RECURSOS QUE NECESITO ACTUALMENTE:
        {faltantes}

        CRITERIOS PARA EXTRAER INFORMACIÓN:
        - OFRECE: recursos que el remitente dice explícitamente que puede dar.
        - QUIERE: recursos que el remitente pide explícitamente.
        - Si no hay una lista clara o hay ambigüedad → "no claro".

        CRITERIOS DE INTERÉS:
        - ALTO: ofrece claramente al menos un recurso que está en mis faltantes.
        - MEDIO: hay indicios de valor pero faltan detalles o el encaje es parcial.
        - BAJO: no ofrece nada útil o el mensaje es confuso.

        CRITERIOS PARA RESPONDER:
        - SI: si el interés es ALTO o MEDIO y el mensaje es mínimamente entendible.
        - NO: si el interés es BAJO o la información es insuficiente.

        FORMATO DE RESPUESTA (OBLIGATORIO, EXACTO):
        OFRECE: <recursos o "no claro">
        QUIERE: <recursos o "no claro">
        INTERES: ALTO / MEDIO / BAJO
        RESPONDER: SI / NO
        RAZON: <máx. 2 frases, sin opiniones personales>

        AUTO-VERIFICACIÓN FINAL (OBLIGATORIA):
        - ¿Usé solo información presente en la carta?
        - ¿Usé "no claro" cuando faltó evidencia?
        - ¿Entregué EXACTAMENTE 5 líneas con las etiquetas correctas?
        Si alguna respuesta es "no", corrige antes de entregar.
        """
    return preguntar_llm(prompt)


def generar_respuesta(carta, sobrantes):
    prompt = f"""
        [ROLE: SYSTEM]
        Eres un jugador humano en un juego de trueque.
        Escribes mensajes breves, naturales y educados, como una persona real.

        OBJETIVO:
        Responder a la carta de forma cordial para explorar un posible intercambio,
        sin comprometerte a un trato cerrado.

        REGLAS ESTRICTAS:
        - SOLO puedes mencionar recursos que estén en "RECURSOS QUE TE SOBRAN".
        - NO inventes recursos ni cantidades.
        - NO prometas intercambios definitivos.
        - NO presiones ni exijas.
        - NO menciones reglas del juego ni razonamientos internos.

        CARTA ORIGINAL:
        Remitente: {carta['remi']}
        Asunto: {carta['asunto']}
        Contenido:
        {carta['cuerpo']}

        RECURSOS QUE TE SOBRAN (ÚNICA FUENTE VÁLIDA):
        {sobrantes}

        ESTILO DEL MENSAJE:
        - Tono amable, humano y respetuoso.
        - Lenguaje sencillo, sin tecnicismos.
        - Puede hacer una pregunta o propuesta abierta.
        - Máximo 5 líneas de texto.
        - Sin listas, sin encabezados, sin firmas largas.

        ANTES DE ENTREGAR:
        - ¿Mencioné solo recursos de la lista?
        - ¿Evité prometer un intercambio cerrado?
        - ¿El texto suena a una persona real?

        ESCRIBE LA RESPUESTA FINAL:
        """
    return preguntar_llm(prompt)

def elegir_intercambio_1x1(sobrantes, faltantes):
    """
    Devuelve (ofrezco, pido) o (None, None) si no se puede.
    Regla simple: elige el primer sobrante y el primer faltante.
    """
    if not sobrantes or not faltantes:
        return None, None

    # Elegimos 1 que sobra y 1 que falta
    ofrezco = next(iter(sobrantes.keys()))
    pido = next(iter(faltantes.keys()))

    return ofrezco, pido

def crear_mensaje_oferta_1x1(ofrezco, pido):
    return (
        f"Hola 🙂\n"
        f"¿Te interesa cambiar 1 de {ofrezco} por 1 de {pido}?\n"
        f"Si te encaja, dime y lo hacemos."
    )


def crear_respuesta_1x1(carta, sobrantes, faltantes):
    ofrezco, pido = elegir_intercambio_1x1(sobrantes, faltantes)
    if not ofrezco or not pido:
        return "Hola 🙂 Gracias por tu mensaje. Ahora mismo no puedo proponer un cambio 1x1 claro. ¿Qué necesitas y qué ofreces exactamente?"

    return (
        f"Hola 🙂 gracias por escribir.\n"
        f"Yo podría cambiar 1 de {ofrezco} por 1 de {pido}.\n"
        f"¿Te encaja?"
    )

def puedo_escribir_a(dest, ahora=None):
    """
    Devuelve True si ya pasó el cooldown y puedo enviarle otra carta a 'dest'.
    """
    if ahora is None:
        ahora = time.time()

    ultimo = ULTIMO_ENVIO_A.get(dest)
    if ultimo is None:
        return True

    return (ahora - ultimo) >= COOLDOWN_SEGUNDOS


def marcar_envio(dest, ahora=None):
    """
    Guarda que acabo de enviarle una carta a 'dest'.
    """
    if ahora is None:
        ahora = time.time()
    ULTIMO_ENVIO_A[dest] = ahora


def limpiar_registro_antiguo(ahora=None):
    """
    Limpia entradas viejas para que el diccionario no crezca sin límite.
    """
    if ahora is None:
        ahora = time.time()

    # Si algo es más viejo que 10x el cooldown, lo borramos
    umbral = ahora - (COOLDOWN_SEGUNDOS * 10)
    for dest, ts in list(ULTIMO_ENVIO_A.items()):
        if ts < umbral:
            del ULTIMO_ENVIO_A[dest]


def enviar_paquete(dest, items):
    """
    Envía un paquete a dest. items es un dict, ejemplo: {"queso": 1}
    """
    try:
        response = httpx.post(
            f"{BASE_URL}/paquete/{dest}",
            json=items,
            timeout=5.0
        )
        response.raise_for_status()
        print(f"Paquete enviado a {dest}: {items}")
        return True
    except Exception as e:
        print(f"Error al enviar paquete a {dest}:", e)
        return False

def carta_parece_aceptacion(carta):
    texto = (carta.get("cuerpo", "") + " " + carta.get("asunto", "")).lower()
    palabras = ["acepto", "me encaja", "trato", "de acuerdo", "ok", "vale", "perfecto"]
    return any(p in texto for p in palabras)




# tarea: SI NOS ESCRIBE SISTEMA: NO ESCRIBIRLE DE VUELTA
# tarea: REALIZAR EL INTERCAMBIO

# ---------------------------
# Main loop
# ---------------------------
def main():
    while True:
        # 1) Obtener estado del juego
        recursos, objetivos, mi_alias, buzon = get_info()
        if recursos is None:
            time.sleep(2)
            continue

        sobrantes = recursos_que_me_sobran(recursos, objetivos)
        faltantes = recursos_que_me_faltan(recursos, objetivos)

        print("=== ESTADO DEL JUEGO ===")
        print("Recursos que tengo:", recursos)
        print("Recursos que necesito:", objetivos)
        print("Recursos que me sobran:", sobrantes)
        print("Recursos que me faltan:", faltantes)

        # 2) Revisar buzón
        utiles = cartas_utiles(buzon, faltantes)

        print("\nCartas que podrían interesarme:")
        if not utiles:
            print("  (ninguna)")
        else:
            for carta in utiles:
                print(f"De: {carta['remi']} | Asunto: {carta['asunto']}")
                print(f"Contenido: {carta['cuerpo']}\n")

                # (opcional) Analizar con LLM
                print("\n--- Analizando carta con LLM ---")
                interpretacion = interpretar_carta(carta, faltantes)
                print(interpretacion)

                # 2.1) Generar respuesta (tu versión 1x1)
                respuesta = crear_respuesta_1x1(carta, sobrantes, faltantes)

                print("\n--- Respuesta generada ---")
                print(respuesta)

                # 2.2) Enviar respuesta
                enviar_carta(
                    remi=mi_alias,
                    dest=carta["remi"],
                    asunto="Re: " + carta["asunto"],
                    cuerpo=respuesta
                )

                # 2.3) NUEVO: si parece aceptación, envío paquete 1x1
                if carta_parece_aceptacion(carta):
                    carta_id = carta.get("id", "")

                    if carta_id and carta_id in PAQUETES_ENVIADOS:
                        print("(paquete) Ya envié paquete para esta carta, no repito.")
                    else:
                        ofrezco, pido = elegir_intercambio_1x1(sobrantes, faltantes)

                        if not ofrezco or not pido:
                            print("(paquete) No puedo hacer 1x1 ahora mismo.")
                        else:
                            # Verificación de que realmente tengo al menos 1 para enviar
                            if recursos.get(ofrezco, 0) >= 1:
                                ok = enviar_paquete(carta["remi"], {ofrezco: 1})
                                if ok and carta_id:
                                    PAQUETES_ENVIADOS.add(carta_id)

                                # Mensaje de seguimiento (opcional)
                                enviar_carta(
                                    remi=mi_alias,
                                    dest=carta["remi"],
                                    asunto="Envío realizado",
                                    cuerpo=f"¡Listo! Te envié 1 de {ofrezco}. Cuando puedas, envíame 1 de {pido} 🙂"
                                )
                            else:
                                print(f"(paquete) Parece que no tengo {ofrezco} para enviar ahora.")

                # 2.4) MUY IMPORTANTE: borrar carta tras procesarla
                borrar_carta(carta["id"])

        # 3) Eliminar cartas inútiles restantes
        print("\nEliminando cartas inútiles...")
        for _, carta in buzon.items():
            if carta not in utiles:
                borrar_carta(carta["id"])

        # 4) Tomar iniciativa SOLO si nadie me ha escrito
        if sobrantes and faltantes and not utiles and not buzon:
            gente = get_gente()
            destinatarios = [
                p["alias"] for p in gente
                if p.get("alias") != mi_alias
            ]

            # Decide oferta 1x1
            ofrezco, pido = elegir_intercambio_1x1(sobrantes, faltantes)
            if ofrezco and pido:
                cuerpo = crear_mensaje_oferta_1x1(ofrezco, pido)

                ahora = time.time()
                limpiar_registro_antiguo(ahora)

                for dest in destinatarios:
                    if not puedo_escribir_a(dest, ahora):
                        print(f"(anti-spam) No escribo a {dest} todavía (cooldown).")
                        continue

                    enviar_carta(
                        remi=mi_alias,
                        dest=dest,
                        asunto="Propuesta de trueque 1x1",
                        cuerpo=cuerpo
                    )
                    marcar_envio(dest, ahora)

        print("\n------------------------\n")
        time.sleep(8)



if __name__ == "__main__":
    main()
