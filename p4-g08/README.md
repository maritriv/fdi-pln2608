# Quijote App (PLN) - Recuperación literal de pasajes

Aplicación CLI en Python para consultar **pasajes literales** de *El ingenioso hidalgo don Quijote de la Mancha* a partir del HTML local (Project Gutenberg).

La app no genera respuestas libres: indexa el corpus y devuelve fragmentos textuales reales donde aparece la consulta.

## 1. Objetivo

- Cargar corpus desde `ZIP` o `HTML`.
- Limpiar y segmentar en pasajes (principalmente párrafos), conservando capítulo cuando se detecta.
- Indexar texto normalizado para búsqueda robusta.
- Mostrar resultados con ranking y contexto legible en terminal.

## 2. Tecnologías usadas

- `BeautifulSoup4` para parseo/limpieza HTML.
- `pathlib` para rutas.
- `Typer` para CLI.
- `joblib` para caché persistente del índice.
- `pytest` para tests básicos.

## 3. Estructura del proyecto

```text
c:\fdi_pln2608_p4
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── quijote_app/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── corpus.py
│   ├── indexing.py
│   ├── models.py
│   ├── search.py
│   └── utils.py
├── data/
├── cache/
└── tests/
    ├── test_corpus.py
    ├── test_normalization.py
    └── test_search.py
```

## 4. Instalación

1. Crear entorno virtual (opcional, recomendado).
2. Instalar dependencias.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

O en modo paquete editable:

```bash
pip install -e .
```

## 5. Datos (corpus)

La app busca por defecto, en este orden:

1. `data/El_ingenioso_hidalgo_don_Quijote_de_la_Mancha.zip`
2. `El_ingenioso_hidalgo_don_Quijote_de_la_Mancha.zip`
3. `data/2000-h.htm`
4. `2000-h.htm`

También puedes indicar `--source`.

Si el ZIP contiene varios HTML, selecciona automáticamente el principal por heurística (nombre + tamaño).

## 6. Uso CLI

Arranque más simple (sin parámetros): abre directamente la sesión interactiva.

```bash
python -m quijote_app
```

La salida usa paneles/tablas en terminal (estilo visual similar a la ayuda de Typer).

### 6.1 Indexar

```bash
python -m quijote_app index --source El_ingenioso_hidalgo_don_Quijote_de_la_Mancha.zip
```

Forzar reindexado:

```bash
python -m quijote_app index --source El_ingenioso_hidalgo_don_Quijote_de_la_Mancha.zip --force
```

### 6.2 Buscar

```bash
python -m quijote_app search dulcinea --limit 5
python -m quijote_app search "dulcinea del toboso" --limit 5
```

Filtro por capítulo:

```bash
python -m quijote_app search dulcinea --chapter "capítulo xiii" --limit 3
```

### 6.3 Modo interactivo (sesión continua)

Lanza una sesión y realiza varias consultas seguidas hasta escribir `exit`:

```bash
python -m quijote_app interactive
python -m quijote_app interactive --source El_ingenioso_hidalgo_don_Quijote_de_la_Mancha.zip
```

Dentro de la sesión:

- escribe una consulta y pulsa Enter;
- `exit`, `quit` o `salir` para cerrar;
- `/help` para ver comandos;
- `/limit 10` para cambiar el límite;
- `/chapter capítulo xxv` para filtrar por capítulo;
- `/chapter off` para quitar el filtro;
- `/stats` para ver estado de la sesión.

### 6.4 Estadísticas

```bash
python -m quijote_app stats
```

### 6.5 Listado de capítulos detectados

```bash
python -m quijote_app chapters --limit 30
```

## 7. Cómo funciona la recuperación

## 7.1 Normalización

Se construye `texto_normalizado` para buscar:

- minúsculas
- eliminación de tildes
- colapso de espacios
- eliminación de puntuación irrelevante

Se conserva `texto_original` para mostrar salida literal.

## 7.2 Ranking

Ranking simple y defendible:

- bonus fuerte por coincidencia exacta de la frase completa;
- bonus por cobertura de términos de la consulta;
- bonus por frecuencia de términos en el pasaje;
- ligera penalización a pasajes muy largos.

## 7.3 Salida

Cada resultado muestra:

- número de resultado
- capítulo detectado (si existe)
- score
- pasaje literal con la coincidencia resaltada con corchetes `[...]`

## 8. Manejo de errores

La CLI controla casos como:

- ruta inexistente;
- ZIP sin HTML;
- corpus vacío o sin pasajes útiles;
- consulta vacía;
- error de caché.

## 9. Tests

Ejecutar:

```bash
pytest
```

Incluye pruebas para:

- normalización y resaltado;
- búsqueda por término y frase;
- carga mínima desde ZIP y segmentación HTML.

## 10. Limitaciones

- La búsqueda es literal/normalizada, no semántica.
- La detección de capítulos depende de la estructura del HTML.
- Consultas muy genéricas pueden devolver muchos resultados similares.

## 11. Mejoras futuras

- operadores booleanos (`AND`/`OR`);
- exportar resultados a `json` o `txt`;
- filtros más finos por rangos de capítulos;
- TUI opcional separada (por ejemplo con Textual).
