import time
import httpx

BASE_URL = "http://147.96.81.252:7719"
#qwen3-vl:8b


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
    """
    Revisa el buzón y devuelve solo las cartas que ofrecen recursos que me faltan
    """
    utiles = []

    for carta_id, carta in buzon.items():
        cuerpo = carta.get("cuerpo", "").lower()  # contenido de la carta
        for recurso in faltantes:
            # Si la carta menciona un recurso que me falta
            if recurso.lower() in cuerpo:
                utiles.append(carta)
                break  # no hace falta mirar otros recursos

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
        Eres un analizador de mensajes para un juego de trueque.
        Tu tarea NO es negociar ni redactar respuestas, solo analizar.

        CONTEXTO DEL JUEGO:
        - Los jugadores intercambian recursos.
        - El objetivo es conseguir los recursos necesarios lo antes posible.

        CARTA RECIBIDA:
        Remitente: {carta['remi']}
        Asunto: {carta['asunto']}
        Contenido:
        {carta['cuerpo']}

        RECURSOS QUE NECESITO ACTUALMENTE:
        {faltantes}

        INSTRUCCIONES IMPORTANTES:
        - No inventes información que no esté en la carta.
        - Si algo no está claro, dilo explícitamente.
        - No escribas texto adicional fuera del formato pedido.

        FORMATO DE RESPUESTA (OBLIGATORIO):
        OFRECE: <recursos o "no claro">
        QUIERE: <recursos o "no claro">
        INTERES: ALTO / MEDIO / BAJO
        RESPONDER: SI / NO
        RAZON: <máx. 2 frases>
        """
    return preguntar_llm(prompt)


def generar_respuesta(carta, sobrantes):
    prompt = f"""
        Eres un jugador humano, educado y estratégico en un juego de trueque.
        Tu tarea es redactar una respuesta breve y natural.

        CONTEXTO:
        - Solo puedes ofrecer recursos que realmente te sobran.
        - No debes prometer intercambios cerrados, solo proponer.
        - El tono debe ser cordial, no agresivo.

        CARTA ORIGINAL:
        Remitente: {carta['remi']}
        Asunto: {carta['asunto']}
        Contenido:
        {carta['cuerpo']}

        RECURSOS QUE TE SOBRAN:
        {sobrantes}

        INSTRUCCIONES:
        - No inventes recursos.
        - No menciones recursos que no estén en la lista.
        - Máximo 5 líneas.
        - Lenguaje claro y humano.

        ESCRIBE LA RESPUESTA:
        """
    return preguntar_llm(prompt)



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

                # Analizar con LLM
                print("\n--- Analizando carta con LLM ---")
                interpretacion = interpretar_carta(carta, faltantes)
                print(interpretacion)

                # Generar respuesta personalizada
                respuesta = generar_respuesta(carta, sobrantes)
                print("\n--- Respuesta generada ---")
                print(respuesta)

                # Enviar respuesta
                enviar_carta(
                    remi=mi_alias,
                    dest=carta["remi"],
                    asunto="Re: " + carta["asunto"],
                    cuerpo=respuesta
                )

                # MUY IMPORTANTE: borrar carta tras procesarla
                borrar_carta(carta["id"])

        # 3) Eliminar cartas inútiles restantes
        print("\nEliminando cartas inútiles...")
        for _, carta in buzon.items():
            if carta not in utiles:
                borrar_carta(carta["id"])

        # 4) Tomar iniciativa SOLO si nadie me ha escrito
        if sobrantes and not utiles:
            gente = get_gente()
            destinatarios = [
                p["alias"] for p in gente
                if p.get("alias") != mi_alias
            ]

            for dest in destinatarios:
                cuerpo = "Hola, tengo estos recursos de sobra:\n"
                for r, c in sobrantes.items():
                    cuerpo += f"- {r}: {c}\n"

                enviar_carta(
                    remi=mi_alias,
                    dest=dest,
                    asunto="Oferta de trueque",
                    cuerpo=cuerpo
                )

        print("\n------------------------\n")
        time.sleep(8)



if __name__ == "__main__":
    main()
