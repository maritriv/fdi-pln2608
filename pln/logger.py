def log(msg: str) -> None:
    """
    Imprime un mensaje formateado por consola con el prefijo del bot.

    Parámetros:
        msg (str): Texto que se desea mostrar en la salida estándar.

    Se utiliza para centralizar el logging del bot y facilitar
    la trazabilidad del flujo de ejecución.
    """
    print(f"[BOT] {msg}")