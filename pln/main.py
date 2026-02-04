import time
import httpx

BASE_URL = "http://147.96.81.252:7719"



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
        aliases = data.get("Alias", ["Desconocido"])
        #alias_mio = aliases[0]

        buzon = data.get("Buzon", {})

        return recursos, objetivos, buzon

    except Exception as e:
        print("Error al obtener info:", e)
        return None, None, None


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


# ---------------------------
# Main loop
# ---------------------------
def main():
    while True:
        recursos, objetivos, buzon = get_info()
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

        # Revisar buzón
        utiles = cartas_utiles(buzon, faltantes)
        print("\nCartas que podrían interesarme:")
        if not utiles:
            print("  (ninguna)")
        else:
            for carta in utiles:
                print(f"De: {carta['remi']} | Asunto: {carta['asunto']}")
                print(f"Contenido: {carta['cuerpo']}\n")

        print("\nEl resto de cartas son inútiles, por lo que procederemos a eliminarlas...")
        # Eliminar cartas inútiles
        for carta_id, carta in buzon.items():
            if carta not in utiles:
                borrar_carta(carta['id'])

        print("\n------------------------\n")
        time.sleep(5)

''' def main():
    while True:
        recursos, objetivos, mi_alias = get_info()
        if recursos is None:
            time.sleep(2)
            continue

        sobrantes = recursos_que_me_sobran(recursos, objetivos)
        print(f"Recursos que me sobran: {sobrantes}")

        if sobrantes:
            gente = get_gente()
            # quitamos nuestro propio alias
            destinatarios = [p for p in gente if p != mi_alias]

            for dest in destinatarios:
                cuerpo = "Hola, tengo estos recursos de sobra:\n"
                for r, c in sobrantes.items():
                    cuerpo += f"- {r}: {c}\n"

                enviar_carta(
                    remi=mi_alias,
                    dest=dest,
                    asunto="Oferta de trueque a 1km de ti",
                    cuerpo=cuerpo
                )

        print("\n------------------------\n")
        time.sleep(10)  # espera 10 segundos entre iteraciones'''



if __name__ == "__main__":
    main()
