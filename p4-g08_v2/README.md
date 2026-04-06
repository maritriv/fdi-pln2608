# 📖 Práctica 4: Quijote App (Buscador Multimodelo)
**Procesamiento del Lenguaje Natural (FDI)** **Versión:** `v1.0` (Release Final)

¡Hola! Esta es nuestra aplicación de terminal para la Práctica 4. Hemos creado un buscador interactivo para *El ingenioso hidalgo don Quijote de la Mancha* que no solo busca palabras exactas, sino que también entiende el significado de las frases y es capaz de responder preguntas usando Inteligencia Artificial.

## 👥 Integrantes del equipo
- Carlota Salazar Martín
- Marina Triviño de las Heras

---

## 🧠 ¿Cómo hemos pensado y montado la práctica?

Al principio empezamos separando la búsqueda clásica de la semántica en versiones distintas, pero nos dimos cuenta de que lo mejor y más limpio era **unirlo todo en una sola aplicación interactiva**. Hemos ido construyendo la app paso a paso:

### Paso 1: Preparar el texto (Chunks y Overlap)
Sabíamos que no podíamos buscar en el libro párrafo por párrafo porque algunos son inmensos y otros muy cortos. Tampoco podíamos cortarlo a lo loco, porque si partíamos una frase por la mitad, perdíamos el contexto. 
**Nuestra solución:** Hemos limpiado el HTML (quitando las cabeceras de Gutenberg) y hemos dividido el texto en bloques ("chunks") de unas 120 palabras. Además, les hemos puesto un *overlap* (solapamiento) de 35 palabras. Así, el final de un bloque es el principio del siguiente y nos aseguramos de no perder ninguna idea importante que caiga justo en el corte.

### Paso 2: Búsqueda Clásica (TF-IDF y Lemas)
Si un usuario busca "gigantes", queríamos que el programa también encontrara "gigante". 
Para esto hemos usado la librería `spaCy`, que nos saca los "lemas" (la raíz de las palabras) y nos limpia las palabras vacías (*stopwords*). 
**¿Cómo ordenamos los resultados?** Hemos programado una fórmula de TF-IDF. Básicamente, si la palabra que buscas aparece muchas veces en un pasaje corto, gana puntos. Pero si es una palabra rarísima en el resto del libro, gana aún más puntos. También damos puntos extra si las palabras están cerquita unas de otras.

### Paso 3: Búsqueda Semántica (Entendiendo el significado)
Queríamos que si alguien busca *"perder la razón por leer demasiados libros"*, el buscador encuentre al Quijote aunque no hayamos escrito la palabra "loco". 
Para hacer esto usamos **Ollama**. Hemos pasado cada uno de los más de 3.000 fragmentos del libro por un modelo de IA (`nomic-embed-text`) que los convierte en listas de números (vectores o *embeddings*). Cuando el usuario busca algo, convertimos su frase en otro vector y medimos la "similitud del coseno" (básicamente, el ángulo matemático entre ellos) para encontrar los textos que más se parecen en concepto.

### Paso 4: RAG (El Oráculo)
Para rematar, hemos añadido la generación de respuestas. En lugar de dejar que la IA se invente las cosas, usamos el buscador semántico para encontrar los 3 o 4 mejores pasajes sobre la pregunta del usuario. Luego, le pasamos esos textos al modelo `llama3` con una instrucción estricta: *"Eres un experto, responde a la pregunta usando ÚNICAMENTE estos textos y cita el capítulo"*. ¡Es como dejarle hacer un examen a la IA pero a libro abierto!

---

## 🛠️ Librerías utilizadas
Hemos respetado estrictamente las herramientas permitidas:
* **`typer` y `rich`**: Las hemos usado para crear una terminal interactiva súper visual, con colores, tablas y paneles, para que sea muy fácil y agradable de usar.
* **`spacy`**: Para entender el español, sacar los lemas y quitar stopwords.
* **`ollama`**: Nuestra conexión con la IA local para sacar los embeddings y generar las respuestas del RAG.

---

## ⚙️ Cómo instalar y preparar todo

Es necesario usar **Python 3.10 o superior** (nosotras recomendamos **3.11** para evitar problemas al compilar las librerías de NLP).

**1. Instalar las dependencias con `uv`:**
```bash
uv sync
```

**2. Descargar el diccionario de español (por si spaCy no lo tiene):**
```bash
uv run python -m spacy info es_core_news_sm --url
uv pip install "<URL_DEVUELTA_POR_EL_COMANDO_ANTERIOR>"
```

**3. Preparar Ollama:**

Ollama debe estar **instalado y en ejecución** para poder usar la búsqueda semántica y el modo RAG.

Como recomendación, en otra terminal diferente:

**3.1. Descargar ollama en caso de no tenerlo:**

En Windows:
```bash
irm https://ollama.com/install.ps1 | iex
```

En Linux/Mac:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Para comprobar que está bien instalado, cierra la terminal y ejecuta:
```bash
ollama
```

**3.2 Descargar los modelos necesarios:**
```bash
ollama pull nomic-embed-text
ollama pull llama3
```

**3.3. Arrancar ollama en caso de que no esté ejecutando anteriormente:**

En muchos sistemas Ollama se inicia automáticamente.

Puedes comprobarlo con:
```bash
ollama list
```

Si no está activo, arráncalo con:
```bash
ollama serve
```
Si aparece un error indicando que el puerto ya está en uso, significa que Ollama ya está ejecutándose correctamente.

```

---

## 🚀 Cómo usar nuestra aplicación

El programa lee automáticamente el `.zip` del Quijote que está en la carpeta `data/`.

**⚠️ ATENCIÓN**: La primera vez que arranques el programa, tardará unos minutos porque tiene que mandar todo el libro a Ollama para calcular los embeddings. Después de eso, los guarda en una caché (`cache/`) y ya será instantáneo.

Arranca la interfaz interactiva con el comando que pide la práctica:
```bash
uv run fdi-pln-2608-p4
```

*(Si alguna vez necesitas que vuelva a pensar todo desde cero y borre la caché, arráncalo con ```uv run fdi-pln-2608-p4 interactive --rebuild```)*


**🎮 Comandos de la consola interactiva:**
Una vez dentro, puedes usar estos comandos:

- Escribe tu consulta directamente (ej. `"dulcinea del toboso"`).

- `/mode classic`: Busca palabras literales (por defecto).

- `/mode semantic`: Busca por significado/concepto usando la IA.

- `/mode rag`: Le haces una pregunta y la IA te responde redactando un texto.

- `/chapter [nombre]`: Para buscar solo en un capítulo (ej. `/chapter capitulo viii`). Para quitarlo pon `/chapter off`.

- `/limit 5`: Cambia cuántos resultados quieres ver.

- `/stats`: Te muestra un resumen de cuántos capítulos hay, el límite actual, etc.

- `exit`: Para cerrar el programa.


---


---

## ❗ Problemas frecuentes

Guía por si ocurre algún problema:

### `ollama` no se reconoce
- Asegúrate de haberlo instalado correctamente
- Reinicia la terminal tras la instalación

### Error con `ollama serve`
- Si indica que el puerto está en uso, Ollama ya está funcionando

### El modo semántico o RAG no funciona
Comprueba:
1. `ollama list`
2. que los modelos están descargados
3. que Ollama está activo

--- 

## 📦 Entrega

El código está formateado con:

```bash
uv format --check
```

Y el ejecutable `.whl` que se entrega en el Campus se ha generado correctamente con el comando `uv build`.
