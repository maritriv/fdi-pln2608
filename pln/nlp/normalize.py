import re

def norm_alias(a):
    return (a or "").strip()

def es_mi_alias(alias, mi_alias):
    return norm_alias(alias).lower() == norm_alias(mi_alias).lower()

def extraer_mi_alias_desde_info(data):
    alias_field = data.get("Alias", None)

    if isinstance(alias_field, str):
        a = alias_field.strip()
        return a if a else "Desconocido"

    if isinstance(alias_field, list):
        if not alias_field:
            return "Desconocido"
        for x in alias_field:
            if isinstance(x, str) and x.strip():
                return x.strip()
        return "Desconocido"

    return "Desconocido"

def normalizar_recurso(nombre):
    if not nombre:
        return ""
    txt = str(nombre).strip().lower()
    txt = re.sub(r"[^a-záéíóúñü]+", "", txt)
    if txt.endswith("s") and len(txt) > 3:
        txt = txt[:-1]
    return txt

def normalizar_texto_libre(s):
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-záéíóúñü0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def es_carta_del_sistema(carta):
    remi = (carta.get("remi", "") or "").strip().lower()
    return remi in {"sistema", "system", "admin"}