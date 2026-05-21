"""Genera plantillas nuevas: 6 frases por JSON, mínimo 20 palabras."""

from pathlib import Path

from scripts.templates import crear_jsons_anotacion, tokenizar_palabras

N_JSON = 13
FRASES_POR_JSON = 6
MIN_PALABRAS = 20
SEED = 46

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]

    info = crear_jsons_anotacion(
        archivo_entrada=root / "corpus_original" / "alice_in_wonderland.txt",
        directorio_salida=root / "asignaciones" / "generadas_6frases",
        tokenizar=tokenizar_palabras,
        granularidad="palabra",
        n_frases=None,
        n_json=N_JSON,
        frases_por_json=FRASES_POR_JSON,
        min_palabras=MIN_PALABRAS,
        seed=SEED,
    )

    print(
        f"Generados {info['n_json']} JSON en {info['directorio']} "
        f"({info['n_frases']} frases, {info['frases_por_json']}/JSON, "
        f"min {info['min_palabras']} palabras)"
    )