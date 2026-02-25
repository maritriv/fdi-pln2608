import re, json
from pln.nlp.normalize import normalizar_recurso

def parsear_json_llm(texto):
    if not texto:
        return {}
    texto = texto.strip()

    try:
        return json.loads(texto)
    except Exception:
        pass

    try:
        ini = texto.find("{")
        fin = texto.rfind("}")
        if ini != -1 and fin != -1 and fin > ini:
            return json.loads(texto[ini:fin + 1])
    except Exception:
        return {}

    return {}

def parse_oferta_v1(cuerpo):
    if not cuerpo or "[OFERTA_V1]" not in cuerpo:
        return [], []

    m_quiero = re.search(r"quiero\s*=\s*(\{.*?\})", cuerpo)
    m_ofrezco = re.search(r"ofrezco\s*=\s*(\{.*?\})", cuerpo)

    quiere, ofrece = [], []
    try:
        if m_quiero:
            d = json.loads(m_quiero.group(1))
            quiere = [normalizar_recurso(k) for k in d.keys()]
        if m_ofrezco:
            d = json.loads(m_ofrezco.group(1))
            ofrece = [normalizar_recurso(k) for k in d.keys()]
    except Exception:
        return [], []

    return quiere, ofrece

def extraer_oferta_1x1_regex(texto):
    if not texto:
        return None, None

    t = " ".join(texto.lower().split())
    patrones = [
        r"te\s+doy\s+1\s+(\w+)\s+a\s+cambio\s+de\s+1\s+(\w+)",
        r"ofrezco\s*1\s+(\w+)\s+por\s+1\s+(\w+)",
        r"cambio\s+1\s+(\w+)\s+por\s+1\s+(\w+)",
        r"1\s+(\w+)\s+por\s+1\s+(\w+)",
    ]
    for pat in patrones:
        m = re.search(pat, t)
        if m:
            return normalizar_recurso(m.group(1)), normalizar_recurso(m.group(2))
    return None, None