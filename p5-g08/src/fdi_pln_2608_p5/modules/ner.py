import re


STOPWORDS = {
    "The", "A", "An", "And", "But", "Or", "If", "Then", "When", "While",
    "It", "He", "She", "They", "We", "I", "You", "This", "That",
}


def extract_named_entities(text):
    """Extrae entidades nombradas simples usando patrones de mayúsculas.

    Es una aproximación ligera útil para el corpus de Alice, donde muchos
    personajes y lugares aparecen como nombres propios: Alice, Queen, King,
    White Rabbit, Mock Turtle, etc.
    """
    pattern = r"\b(?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+)*\b"
    candidates = re.findall(pattern, text)

    entities = []
    seen = set()

    for candidate in candidates:
        if candidate in STOPWORDS:
            continue
        if candidate not in seen:
            entities.append(candidate)
            seen.add(candidate)

    return entities