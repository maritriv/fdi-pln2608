"""Funciones de normalización y comparación textual.

Este módulo contiene utilidades para:
- normalizar alias de agentes,
- comparar nombres de agentes ignorando mayúsculas,
- limpiar nombres de recursos,
- limpiar texto libre,
- detectar mensajes generados por el sistema.

Estas funciones ayudan a hacer el comportamiento del agente más robusto
frente a diferencias de formato en los mensajes.
"""

import re


def norm_alias(alias: str | None) -> str:
    """Normaliza un alias eliminando espacios laterales.

    Si el alias es None, devuelve una cadena vacía.
    """
    return (alias or "").strip()


def es_mi_alias(alias: str | None, mi_alias: str | None) -> bool:
    """Comprueba si dos alias representan al mismo agente.

    La comparación ignora mayúsculas/minúsculas y espacios laterales.
    """
    return norm_alias(alias).lower() == norm_alias(mi_alias).lower()


def extraer_mi_alias_desde_info(data: dict) -> str:
    """Extrae el alias del agente desde la respuesta de Butler.

    Butler puede devolver el alias como:
    - cadena,
    - lista de cadenas,
    - valor vacío.

    En caso de no encontrar un alias válido,
    devuelve el texto 'Desconocido'.
    """
    alias_field = data.get("Alias", None)

    if isinstance(alias_field, str):
        alias = alias_field.strip()
        return alias if alias else "Desconocido"

    if isinstance(alias_field, list):
        if not alias_field:
            return "Desconocido"

        for item in alias_field:
            if isinstance(item, str) and item.strip():
                return item.strip()

        return "Desconocido"

    return "Desconocido"


def normalizar_recurso(nombre: str | None) -> str:
    """Normaliza el nombre de un recurso.

    Operaciones realizadas:
    - pasa a minúsculas,
    - elimina caracteres especiales,
    - elimina plurales simples terminados en 's'.

    Esto facilita comparar recursos escritos de formas distintas.
    """
    if not nombre:
        return ""

    txt = str(nombre).strip().lower()
    txt = re.sub(r"[^a-záéíóúñü]+", "", txt)

    if txt.endswith("s") and len(txt) > 3:
        txt = txt[:-1]

    return txt


def normalizar_texto_libre(texto: str | None) -> str:
    """Normaliza texto libre eliminando ruido textual.

    Se conservan únicamente:
    - letras,
    - números,
    - espacios.

    También se reducen múltiples espacios consecutivos.
    """
    if not texto:
        return ""

    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúñü0-9\s]+", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def es_carta_del_sistema(carta: dict) -> bool:
    """Detecta si una carta fue enviada por el sistema.

    Se consideran mensajes del sistema aquellos cuyo remitente sea:
    - sistema
    - system
    - admin
    """
    remi = (carta.get("remi", "") or "").strip().lower()

    return remi in {"sistema", "system", "admin"}
