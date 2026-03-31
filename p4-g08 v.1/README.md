# fdi-pln-2608-p4 (Practica 4 v1)

Aplicacion de terminal para recuperacion de informacion sobre el Quijote.
Esta version (`v1.0`) implementa busqueda clasica por lemas, sin embeddings.

## Integrantes
- Completar con nombre y apellidos (integrante 1)
- Completar con nombre y apellidos (integrante 2)
- Completar con nombre y apellidos (integrante 3)

## Objetivo de la v1
- Preprocesar el texto fuente en chunks de tamano sustancial con overlap.
- Lematizar corpus y consulta con spaCy.
- Eliminar stopwords para mejorar precision.
- Ordenar resultados por relevancia con ranking TF-IDF.
- Devolver pasajes reales del corpus (no respuestas generadas).

## Estructura
```text
quijote_app/
  __init__.py
  __main__.py
  cli.py
  config.py
  corpus.py
  indexing.py
  models.py
  nlp.py
  search.py
  utils.py
data/
tests/
pyproject.toml
README.md
```

## Dependencias permitidas usadas
- `typer`
- `rich`
- `spacy`

## Instalacion
```bash
uv sync
```

Para mejor lematizacion, instalar el modelo de espanol de spaCy si no esta disponible en el laboratorio:
```bash
uv run python -m spacy download es_core_news_sm
```

## Ejecucion
Comando exigido por la practica:
```bash
uv run fdi-pln-2608-p4
```

Comandos principales:
```bash
uv run fdi-pln-2608-p4 index
uv run fdi-pln-2608-p4 search "dulcinea del toboso" --limit 5
uv run fdi-pln-2608-p4 interactive
uv run fdi-pln-2608-p4 stats
uv run fdi-pln-2608-p4 chapters --limit 30
```

## Datos del corpus
Rutas buscadas por defecto:
1. `data/El_ingenioso_hidalgo_don_Quijote_de_la_Mancha.zip`
2. `El_ingenioso_hidalgo_don_Quijote_de_la_Mancha.zip`
3. `data/2000-h.htm`
4. `2000-h.htm`

## Preprocesado (v1)
- Carga HTML desde ZIP o fichero suelto.
- Limpieza de ruido (cabeceras/pies Gutenberg, lineas no utiles).
- Segmentacion en unidades y creacion de chunks con overlap.
- Normalizacion para matching robusto.
- Lematizacion y filtrado de stopwords en indice y consulta.

## Ranking clasico
Puntuacion combinada de:
- TF-IDF sobre lemas.
- Cobertura de terminos de consulta.
- Bonus por coincidencia exacta de frase/lemmas.
- Bonus por proximidad de terminos.

## Calidad y pruebas
Ejecutar tests:
```bash
uv run pytest
```

Formato (requisito de entrega):
```bash
uv format --check
```

Si vuestra version de `uv` no tiene `format`, usar:
```bash
uv tool run ruff format --check .
```

## Build del wheel
```bash
uv build
```

Wheel esperado:
`dist/fdi_pln_2608_p4-1.0-py3-none-any.whl`

## Entrega
- Release GitHub: `p4v1.0`
- Wheel en campus con nombre `fdi_pln_2608_p4-1.0-py3-none-any.whl`
