# Informe P5 - Grupo 8

## 1. Objetivo y filosofía: Entender

El objetivo de esta práctica era introducirnos en la "caja negra" de los Modelos de Lenguaje para ver cómo funcionan por dentro desde cero. Nuestro objetivo no era entrenar un modelo gigantesco que compitiera con los sistemas actuales, sino entender íntimaente, línea a línea, qué piezas matemáticas y de software hacen falta para construir un pipeline de Procesamiento del Lenguaje Natural funcional.

Para ello, hemos construido dos sistemas interconectados. 
- Primero, un **modelo de lenguaje causal** capaz de generar texto de forma autorregresiva.
- Segundo, un modelo de **Reconocimiento de Entidades Nombradas (NER)** que reutiliza el conocimiento sintáctico aprendido en la primera fase.
Todo el sistema ha sido programado desde cero, entrenado en CPU y encapsulado en una herramienta de línea de comandos completa y fácil de usar.

En este informe, detallamos nuestras decisiones de diseño, los experimentos realizados, y nuestras reflexiones sobre los obstáculos que fuimos superando.

--

## 2. El primer paso: Enseñar a leer (Tokenización BPE)

Un modelo no puede procesar lenguaje si antes no decidimos cómo trocearlo. Una opción es tokenizar por palabras completas, pero que descartamos porque se generaría entonces un vocabulario inmanejable y, además, el modelo no sabría qué hacer con palabras nuevas que entrarar. Otra opción puede ser tokenizar por caracteres individuales, pero entonces las secuencias serían tan largas que el modelo perdería el hilo del significado.

Como en el punto medio está la virtud, nuestra solución fue implementar nuestro propio tokenizador BPE (*Byte Pair Encoding*). Lo entrenamos iterativamente para fusionar los pares de caracteres más frecuentes de los textos de Lewis Carroll (estamos refiriéndonos a: *Alicia en el País de las Maravillas* y *A Través del Espejo*).

**Ejemplo de nuestro BPE:**
Si pasamos la frase *"Alice went to Wonderland"* por nuestro analizador:
* Caracteres originales: 24
* Tokens resultantes: 11
* Ratio de comprensión: 2.18 caracteres por token

> Para comprobarlo empíricamente, podemos introducir esta instrucción en nuestra línea de comandos:
>  ```bash
>  uv run fdi-pln-2608-p5 analyze-bpe --weights checkpoints/p5_causal_2608.pth \
>  --text "Alice went to Wonderland"
>  ```
>  Se pueden probar otras frases sustituyéndolas en el apartado de "text".

Al inspeccionar los tokens, notamos que algunas palabras con mayúsculas se segmentan de forma poco natural. Aquí hemos encontrado una limitación que, sin embargo, era esperable al entrenar un BPE con un corpus tan reducido. Pero matemáticamente cumple su propósito a la perfección: mantener un vocabulario pequeño y poder codificar cualquier texto de entrada.

---

## 3. Construyendo el cerebro: La Arquitectura Transformer

Fuimos implementando la arquitectura bloque a bloque: embeddings posicionales, atención multi-cabezal y capas feed-forward con GELU. Pero si tuviéramos que destacar el aprendizaje más importante de esta fase, fue comprender la necesidad de la **atención escalada**.

La atención es el verdadero motor del Transformer, ya que es el mecanismo que permite al modelo determinar qué palabras de una secuencia son más importantes para cada una de las demás. Para lograrlo, cada posición del texto genera tres representaciones: *Queries* (Q), *Keys* (K) y *Values* (V). Con ellas se calcula la afinidad entre tokens mediante la Atención de Producto Escalar Escalado (*Scaled Dot-Product Attention*):

$$ \text{Atención}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

Al programar esta ecuación nos dimos cuenta de que la teoría tiene una traducción técnica vital. Entendimos el verdadero valor de dividir por la raíz cuadrada de la dimensión (`sqrt(head_dim)` o $\sqrt{d_k}$). Sin este escalado matemático, los productos escalares entre nuestras *Queries* y *Keys* crecían desproporcionadamente; esto saturaba la función Softmax, empujaba los gradientes a cero y paralizaba por completo el aprendizaje de la red en nuestra CPU.


Utilizamos este mismo "cerebro" de dos formas distintas, cambiando solo el flujo de la información:
*   **Para generar texto (Causal):** Aplicamos una máscara para que el modelo solo pudiera "mirar hacia atrás" (matemáticamente, esto es una matriz triangular inferior). Para predecir la siguiente palabra, no puedes hacer trampas mirando el futuro.
*   **Para buscar entidades (NER):** Eliminamos la máscara. Para clasificar si una palabra es un personaje o un lugar, la red necesita un contexto **bidireccional** (leer tanto el inicio como el final de la oración).

---

## 4. Fase 1: Entrenamiento Causal y Experimentación

Diseñamos una configuración deliberadamente pequeña para que los tiempos de entrenamiento en CPU fueran asumibles, aceptando el sacrificio en la capacidad del modelo para memorizar textos largos.

**Nuestra configuración final (`p5_causal_2608.pth`):**
*   `context_size` = 128 | `seq_len` = 128 | `vocab_size` = 300
*   `d_model` = 128 | `n_heads` = 4 | `n_layers` = 4
*   `batch_size` = 64 | `epochs` = 5 | `lr` = 0.0003 | `dropout` = 0.1

### El impacto de la Temperatura y el Top-k

Entrenar el modelo es solo la mitad del trabajo; la otra mitad es saber cómo extraerle el texto. Realizamos varias pruebas cambiando los parámetros de muestreo a partir del prompt `"Alice was"`, limitando la salida a 40 tokens.

Tanto la temperatura como el top-k son mecanismos que intervienen en el último paso del modelo: el muestreo (sampling).

> Cuando tu Transformer procesa un texto, no predice una única palabra ganadora. Lo que hace es generar un vector enorme (los logits) y aplicarles una función Softmax para asignar una probabilidad a cada palabra del vocabulario. Estos dos parámetros alteran esa distribución de probabilidades antes de que "tiremos el dado" para elegir la siguiente palabra.
> Normalmente se usan a la vez porque hacen un equipo perfecto:
> - El Top-k actúa primero como filtro: expulsa a las palabras completamente absurdas para asegurar que la gramática no se rompa.
> - La Temperatura actúa después sobre las candidatas "buenas" que han sobrevivido al filtro: decide si tiramos los dados de forma arriesgada (alta) o conservadora (baja) entre ese grupo de finalistas.

| Temp | Top-k | Texto generado (extracto literal) | Nuestra lectura analítica |
| :---: | :---: | :--- | :--- |
| **0.5** | 10 | *"Alice was a long way to find that they must have to do that!” “i haven’t tell you see, oh, if"* | **Demasiado conservador.** La red elige los caminos más seguros. El resultado es gramaticalmente estable pero repetitivo y poco imaginativo. |
| **0.8** | 10 | *"Alice was a pawn out of it, and she could not thought to herself, “if it would have a good must grine,"* | **El punto dulce.** Permite cierta exploración del vocabulario aportando variedad y "estilo Lewis Carroll", sin destruir por completo la estructura de la oración. |
| **1.2** | 20 | *"Alice was all thank croquet, which the time with so. * * * * *"* | **Colapso del modelo.** Forzar a una red tan pequeña a explorar tokens poco probables (alta temperatura) añade demasiado ruido; pierde el hilo y entra en bucles de puntuación. |

**Conclusión:** Un modelo pequeño no tiene representaciones lingüísticas robustas. Subir la temperatura rompe la coherencia rapidísimo. Por eso, fijamos los valores por defecto de nuestra aplicación en `temperature=0.8` y `top-k=20`.

---

## 5. Fase 2: Reciclando conocimiento (Transfer Learning para NER)

En lugar de entrenar un clasificador NER desde cero, aplicamos Transfer Learning. Cargamos los pesos de nuestro modelo causal (que ya sabe cómo se estructuran las palabras en inglés) y le acoplamos una cabeza lineal nueva para predecir etiquetas BIO (`PER` y `LOC`).

Utilizamos los datos que anotamos a mano en la preentrega junto al Grupo 1, donde aplicamos una clasificación BIO adaptada para distinguir inicios y continuaciones de personajes (`pi`, `pc`) y lugares (`li`, `lc`), convirtiéndolos finalmente de forma automática al estándar CoNLL.

**Resultados de la evaluación (`p5_ner_2608.pth`):**

| Métrica | Token-level | Entity-level |
| :--- | :--- | :--- |
| **Accuracy** | 0.8881 | - |
| **Precision** | 0.2666 | 0.1132 |
| **Recall** | 0.7053 | 0.5909 |
| **F1-Score** | 0.3869 | 0.1901 |

**Desglose por clase:**

| Tipo | Precision | Recall | F1 | Soporte (Ejemplos reales) |
| :--- | :--- | :--- | :--- | :--- |
| **LOC** | 0.1027 | 0.8824 | 0.1840 | 17 |
| **PER** | 0.1168 | 0.5376 | 0.1919 | 93 |


### Análisis de las métricas

La altísima *accuracy* (89%) es un espejismo estadístico provocado por el fuerte desbalance de clases (casi todo el texto es la clase "O"). Las métricas a nivel de entidad cuentan la historia real.

Sin embargo, tener un *Recall* de casi el 60% es un éxito: demuestra que el modelo **sí aprende a localizar entidades**, encontrando con facilidad personajes como *Alice* o el *Hatter*. El problema radica en la precisión (~11%). Al introducir pesos de clase en el entrenamiento para evitar que el modelo ignorase a las minorías, creamos un modelo excesivamente "valiente" que marca demasiados falsos positivos (como preposiciones o palabras de enlace).

### La decisión del Post-procesado

Nos dimos cuenta de que, aunque el modelo encontraba las entidades correctas, la salida en bruto de la inferencia era muy ruidosa y frustrante para un usuario.

Para solucionar esto, añadimos una pequeña capa heurística de post-procesado **solo en el momento de la inferencia final** (no afecta a los pesos ni altera las métricas de evaluación mostradas arriba). Este post-procesado:
1.  Descarta palabras funcionales evidentes (*the*, *ran*, *through*).
2.  Normaliza convenciones del dominio (asegurando que *Queen* o *White Rabbit* se marquen como `PER`, y *Wonderland* como `LOC`).
3.  Agrupa palabras consecutivas en una sola entidad visual (por ejemplo, si la red escupía [`White, PER`] y [`Rabbit, PER`], lo unifica como [`White Rabbit, PER`]).

Gracias a esto, convertimos una predicción ruidosa en una herramienta por terminal (`CLI`) verdaderamente útil y clara, demostrando que en el PLN real, el Deep Learning y las reglas heurísticas a menudo trabajan de la mano.

---

## 6. Ingeniería de Software y Reproducibilidad

Todo este pipeline no tendría valor si solo funcionara en nuestros ordenadores. Invertimos tiempo en asegurar que la práctica fuera un producto completo:

*   **Checkpoints autocontenidos:** Guardamos los pesos, la configuración y el tokenizador BPE en un mismo archivo `.pth`.
*   **Interfaz CLI:** Construimos un menú interactivo con `Typer` y `Rich` que permite generar texto, evaluar o inferir sin tener que memorizar comandos interminables.
*   **Empaquetado:** Todo el proyecto compila de forma limpia en un `.whl` a través de `uv build`.

---

## 7. Conclusiones Finales

Las limitaciones que hemos documentado (deformaciones léxicas, pérdida de coherencia a largo plazo, o falsos positivos en NER) no son fracasos; son la consecuencia matemática esperable de entrenar una arquitectura reducida, con un corpus literario pequeño, en una CPU local. 

Sin embargo, el objetivo didáctico se ha cumplido con creces. Hemos comprobado "manualmente" cómo las representaciones aprendidas por un BPE viajan por los bloques de atención, cómo la temperatura modifica la creatividad de la red, y cómo un modelo generativo puede ser reciclado para extraer información estructurada (NER). La Inteligencia Artificial actual se asienta sobre estas mismas piezas; nosotras, simplemente, hemos construido nuestra propia maqueta a escala.

---
**Integrantes:**
Marina Triviño de las Heras | Carlota Salazar Martín (Grupo 2608)

---------------------------------------------------------------------------------------
En esta práctica hemos construido un pequeño sistema de procesamiento del lenguaje natural basado en Transformers. El trabajo tiene dos partes conectadas entre sí, un modelo de lenguaje causal capaz de generar textodespués y un modelo de reconocimiento de entidades nombradas (NER) que reutiliza el conocimiento aprendido por el modelo causal.

La idea principal no era entrenar un modelo grande ni competir con sistemas actuales, sino entender de forma práctica qué piezas hacen falta para construir un pipeline moderno de NLP: tokenización, embeddings, atención, bloques Transformer, entrenamiento autoregresivo, transferencia de pesos y evaluación de una tarea de clasificación secuencial.

Por eso el proyecto se ha planteado como una implementación pequeña pero completa. El sistema permite:

- entrenar el modelo causal
- generar texto a partir de un prompt
- preparar el corpus NER en formato CoNLL/BIO
- entrenar el modelo NER reutilizando el backbone causal
- evaluar métricas token-level y entity-level
- listar entidades nombradas encontradas en un fichero de texto

Todo queda integrado en un único ejecutable:

```bash
uv run fdi-pln-2608-p5
```

Además del modo por comandos, se ha añadido un menú interactivo con Typer y Rich para que la práctica sea más sencilla e intuitiva.

---

## 2. Organización del proyecto

La estructura general del proyecto es la siguiente:

```text
src/fdi_pln_2608_p5/
|-- cli.py                 # entry point del wheel
|-- main.py                # arranque mínimo de la app Typer
|-- checkpoint.py          # guardado y carga de checkpoints
|-- tokenizer.py           # tokenizador BPE propio
|-- utils.py               # semillas, dispositivo y utilidades 
|
|-- cli_app/               # interfaz Typer + Rich
|-- model/                 # atención, Transformer y modelo NER
|-- data/                  # dataset causal y conversión NER
|-- training/              # entrenamiento causal y NER
|-- evaluation/            # evaluación NER y análisis BPE
`-- generation/            # generación de texto e inferencia NER
```

Esta organización separa claramente las distintas responsabilidades del proyecto: la interfaz de línea de comandos, la lógica de entrenamiento, la generación de texto, la evaluación y los componentes internos del modelo. La idea era mantener una arquitectura modular donde cada parte pudiera evolucionar de forma relativamente independiente.

Además, esta separación facilita bastante la reutilización de componentes, las pruebas individuales de cada módulo y la trazabilidad del pipeline completo, desde la preparación de datos hasta la evaluación final de modelos.


---

## 3. Corpus utilizado

Para el modelo de lenguaje se utilizaron los textos proporcionados para la práctica:

- `resources/alice_in_wonderland.txt`
- `resources/looking_glass.txt`

Ambos textos pertenecen al universo literario de Lewis Carroll y son adecuados para esta práctica porque contienen diálogos, personajes recurrentes, nombres propios frecuentes y referencias a lugares. Esto permite que el modelo causal aprenda ciertos patrones de estilo narrativo y, al mismo tiempo, que la tarea NER tenga sentido dentro del mismo dominio textual.

El corpus no es especialmente grande, y eso condiciona los resultados obtenidos. Aun así, para una implementación didáctica entrenada en CPU resulta suficiente ya que permite entrenar el modelo, generar ejemplos, analizar errores reales y comprobar el funcionamiento completo del pipeline sin que el coste de entrenamiento sea inabordable.

Para NER se reutilizó el corpus anotado durante la preentrega. En la práctica final ya está integrado en:

```text
data/ner/merged.json
data/ner/final.conll

---

## 4. Tokenización BPE

El modelo no trabaja directamente con palabras completas. Se implementó un tokenizador BPE (*Byte Pair Encoding*) propio, entrenado sobre el corpus literario.

La razón de usar BPE es que ofrece un equilibrio entre caracteres y palabras. Si se tokeniza solo por caracteres, las secuencias se vuelven largas y al modelo le cuesta aprender unidades con significado. Si se tokeniza solo por palabras, cualquier palabra no vista se convierte en un problema. BPE permite representar palabras frecuentes con pocos tokens y palabras raras mediante fragmentos más pequeños.

El procedimiento seguido fue:

1. empezar con caracteres individuales;
2. contar pares consecutivos frecuentes;
3. fusionar los pares más comunes;
4. repetir hasta alcanzar el tamaño de vocabulario deseado;
5. reutilizar esas fusiones para codificar nuevos textos.

Un ejemplo real de análisis BPE:

```bash
uv run fdi-pln-2608-p5 analyze-bpe \
  --weights checkpoints/p5_causal_2608.pth \
  --text "Alice went to Wonderland"
```

Resultado resumido:

| Texto | Caracteres | Tokens | Ratio caracteres/token |
| --- | ---: | ---: | ---: |
| `Alice went to Wonderland` | 24 | 11 | 2.18 |

La segmentación obtenida no siempre es intuitiva. Por ejemplo, algunas palabras con mayúsculas o poco frecuentes se rompen en piezas pequeñas. Esto es una limitación esperable al entrenar BPE con un corpus reducido. Aun así, el tokenizador cumple su función: mantiene un vocabulario pequeño y permite codificar cualquier texto de entrada.

---

## 5. Arquitectura Transformer

El núcleo del proyecto es un Transformer pequeño implementado manualmente. La arquitectura incluye:

- embeddings de token;
- embeddings posicionales;
- bloques Transformer pre-norm;
- atención multi-cabezal;
- capas feed-forward con GELU;
- dropout;
- una cabeza final de lenguaje para generación;
- una cabeza BIO para NER.

La atención es el componente más importante. Cada posición de la secuencia genera tres representaciones: queries, keys y values. Con ellas se calcula qué tokens deben atender a qué otros tokens.

Para estabilizar el cálculo se usa atención escalada:

```text
softmax((Q @ K.T) / sqrt(head_dim)) @ V
```

El escalado por `sqrt(head_dim)` evita que los productos internos crezcan demasiado y saturen la softmax. Esta decisión es estándar en Transformers y resulta especialmente importante cuando se entrena con recursos limitados, porque ayuda a que el entrenamiento sea más estable.

El mismo backbone se usa de dos formas:

- en generación, con máscara causal, para impedir que el modelo vea tokens futuros;
- en NER, sin máscara causal, porque para clasificar una palabra interesa usar contexto izquierdo y derecho.

Esta diferencia es importante: la arquitectura base es la misma, pero la tarea cambia la forma de usar la atención.

---

## 6. Modelo causal de generación

El primer modelo entrenado fue el modelo de lenguaje causal. Su objetivo es predecir el siguiente token a partir de los tokens anteriores.

La configuración final fue pequeña para que el entrenamiento fuera viable en CPU:

| Parámetro | Valor |
| --- | ---: |
| `context_size` | 128 |
| `d_model` | 128 |
| `n_heads` | 4 |
| `n_layers` | 4 |
| `dropout` | 0.1 |
| `lr` | 0.0003 |

La decisión de mantener un modelo pequeño fue deliberada. Un modelo mayor podría parecer más ambicioso, pero con un corpus reducido y entrenamiento en CPU no necesariamente iba a generalizar mejor. En las pruebas, aumentar mucho la arquitectura hacía el entrenamiento más lento sin una mejora clara en la calidad de salida.

El checkpoint final es:

```text
checkpoints/p5_causal_2608.pth
```

Ejemplo de uso:

```bash
uv run fdi-pln-2608-p5 generate \
  --weights checkpoints/p5_causal_2608.pth \
  --prompt "Alice was" \
  --max-new-tokens 80 \
  --top-k 20
```

Las generaciones capturan parcialmente el estilo del corpus: aparecen diálogos, puntuación literaria y nombres propios característicos. Sin embargo, también aparecen palabras deformadas, repeticiones y pérdida de coherencia en secuencias largas.

Esto no se interpreta como un fallo total del modelo, sino como una consecuencia razonable del tamaño del corpus y del tamaño del Transformer. El sistema demuestra que el pipeline funciona, aunque no tenga capacidad suficiente para producir texto largo de alta calidad.

---

## 7. Modelo NER

La segunda parte de la práctica consiste en adaptar el modelo causal a reconocimiento de entidades nombradas.

En vez de entrenar un modelo NER completamente nuevo, se reutiliza el backbone Transformer aprendido durante el entrenamiento causal. Se conservan:

- embeddings;
- posiciones;
- bloques Transformer.

La cabeza de lenguaje se sustituye por una capa lineal que predice etiquetas BIO por token.

Esta decisión tiene sentido porque el modelo causal ya ha aprendido cierta información sobre el corpus: patrones de caracteres, nombres frecuentes, estilo y contexto local. Aunque el modelo sea pequeño, reutilizar ese conocimiento es mejor que empezar NER desde cero.

El checkpoint final es:

```text
checkpoints/p5_ner_2608.pth
```

Entrenamiento:

```bash
uv run fdi-pln-2608-p5 train-ner \
  --data data/ner/final.conll \
  --causal-weights checkpoints/p5_causal_2608.pth \
  --output checkpoints/p5_ner_2608.pth
```

Evaluación:

```bash
uv run fdi-pln-2608-p5 eval-ner \
  --weights checkpoints/p5_ner_2608.pth \
  --data data/ner/final.conll
```

Resultados:

| Métrica | Valor |
| --- | ---: |
| Token accuracy | 0.8881 |
| Token precision | 0.2666 |
| Token recall | 0.7053 |
| Token F1 | 0.3869 |
| Entity precision | 0.1132 |
| Entity recall | 0.5909 |
| Entity F1 | 0.1901 |

Desglose por tipo:

| Tipo | Precision | Recall | F1 | Soporte |
| --- | ---: | ---: | ---: | ---: |
| LOC | 0.1027 | 0.8824 | 0.1840 | 17 |
| PER | 0.1168 | 0.5376 | 0.1919 | 93 |

La accuracy por token es alta porque la mayoría de tokens son `O`. Por eso no basta con mirar accuracy. Las métricas entity-level son más honestas: evalúan si el modelo detecta entidades completas y no solo etiquetas aisladas.

El modelo tiene un recall relativamente alto, pero baja precisión. Esto significa que encuentra muchas entidades, pero también tiende a marcar algunas palabras que no deberían ser entidades. La causa principal es el desbalance del corpus: hay muchísimos tokens sin entidad y muy pocos ejemplos positivos, especialmente de localizaciones.

### Postprocesado de inferencia

Durante las pruebas finales observamos que el modelo detectaba entidades relevantes del corpus, pero la salida del comando `ner` todavía era incómoda de leer. Por ejemplo, podían aparecer palabras funcionales como `the`, `ran` o `through` marcadas como entidades, y algunos nombres conocidos del dominio salían con el tipo cambiado.

Para mejorar la experiencia de uso se añadió una fase pequeña de postprocesado después de la predicción del modelo. Esta fase no modifica el entrenamiento, no cambia los pesos y no altera las métricas calculadas sobre el dataset. Solo actúa en el momento de presentar la inferencia al usuario.

El postprocesado hace tres cosas sencillas:

- descarta palabras funcionales que no pueden ser entidades en este dominio;
- normaliza nombres frecuentes de *Alice in Wonderland*, como `Queen`, `Wonderland`, `White Rabbit` o `March Hare`;
- agrupa palabras consecutivas del mismo tipo para mostrar entidades compuestas en una sola fila.

La idea no es esconder las limitaciones del modelo, sino convertir una predicción ruidosa en una salida más clara y útil para probar la herramienta. Es una decisión práctica: el modelo sigue siendo pequeño y el corpus sigue siendo limitado, pero el CLI final resulta mucho más comprensible para quien lo ejecuta.

```bash
uv run fdi-pln-2608-p5 ner \
  --weights checkpoints/p5_ner_2608.pth \
  --file examples/text.txt
```

Ejemplos de entidades detectadas:

| Entidad | Tipo predicho |
| --- | --- |
| Queen | PER |
| Wonderland | LOC |
| White Rabbit | PER |
| Oxford | LOC |
| Hatter | PER |
| March Hare | PER |
| Alice | PER |
| garden | LOC |
| hall | LOC |

Los falsos positivos originales aparecen por el desbalance del dataset y por el tamaño reducido del corpus anotado. Se añadieron pesos de clase para que el modelo no ignorase todas las entidades, pero eso aumentó la tendencia a sobredetectar. El postprocesado final reduce ese ruido en inferencia y presenta una salida más razonable y coherente con el dominio de la práctica.

---

## 8. Experimentos de generación

Además del entrenamiento principal, se realizaron pruebas variando `temperature` y `top-k`.

El objetivo no era encontrar una combinación perfecta, sino observar cómo cambia el comportamiento de un modelo pequeño según el muestreo.

Comando usado:

```bash
uv run fdi-pln-2608-p5 experiment-generate \
  --weights checkpoints/p5_causal_2608.pth \
  --prompt "Alice was" \
  --out reports/generation_experiments.md \
  --max-new-tokens 40
```

Resumen cualitativo:

| Temperature | Top-k | Comportamiento observado |
| ---: | ---: | --- |
| 0.5 | 10 | Más conservador, menos variado y algo repetitivo. |
| 0.8 | 10/20 | Mejor equilibrio entre variedad y control. |
| 1.2 | 20/50 | Más creativo, pero también más inestable. |

La conclusión principal fue que, en un modelo pequeño, subir mucho la temperatura aumenta rápidamente el ruido. El modelo empieza a explorar tokens menos probables, pero como no tiene una representación lingüística muy robusta, aparecen más deformaciones y saltos de coherencia.

Por eso, para ejemplos de entrega, los valores más razonables son `temperature=0.8` y `top-k` entre 10 y 20.

---

## 9. Interfaz de línea de comandos

La práctica se entrega como un CLI único. Esto es importante porque el enunciado pide poder ejecutar la práctica mediante un comando instalado desde el wheel.

El comando principal es:

```bash
uv run fdi-pln-2608-p5
```

Al ejecutarlo sin argumentos se abre un menú interactivo. Este menú permite probar las partes más importantes sin memorizar todos los comandos:

- generación de texto;
- detección de entidades NER;
- evaluación del modelo NER;
- análisis de tokenización BPE;
- ejemplos de comandos directos.

También se conservan los comandos reproducibles:

```bash
uv run fdi-pln-2608-p5 generate --weights checkpoints/p5_causal_2608.pth --prompt "Alice was"
uv run fdi-pln-2608-p5 ner --weights checkpoints/p5_ner_2608.pth --file examples/text.txt
uv run fdi-pln-2608-p5 eval-ner --weights checkpoints/p5_ner_2608.pth --data data/ner/final.conll
```

Se decidió usar Typer y Rich porque permiten una interfaz más clara sin añadir dependencias no permitidas. El CLI final no es solo una colección de scripts: actúa como una pequeña herramienta de laboratorio para entrenar, evaluar y probar el sistema.

---

## 10. Reproducibilidad

Los pasos principales para reproducir la práctica son:

```bash
uv sync
uv format --check
uv run fdi-pln-2608-p5 --help
uv run fdi-pln-2608-p5 generate --weights checkpoints/p5_causal_2608.pth --prompt "Alice was"
uv run fdi-pln-2608-p5 eval-ner --weights checkpoints/p5_ner_2608.pth --data data/ner/final.conll
uv build
```

El wheel generado es:

```text
dist/fdi_pln_2608_p5-1.0-py3-none-any.whl
```

Los checkpoints son autocontenidos: guardan pesos, configuración, métricas y tokenizador. Esto evita depender de configuraciones externas y facilita que el profesor pueda cargar directamente los modelos entregados.

---

## 11. Limitaciones

La limitación más importante es el tamaño del corpus. El modelo causal aprende ciertos patrones de estilo, pero no tiene datos suficientes para generar texto largo y coherente de forma consistente.

En generación aparecen:

- palabras deformadas;
- repeticiones;
- frases que empiezan con estilo literario pero pierden sentido;
- dependencia fuerte de `temperature` y `top-k`.

En NER, la principal dificultad es el desbalance. Hay muchos tokens `O` y pocos tokens de entidad. Además, el número de ejemplos de localizaciones es especialmente bajo. Esto hace que el modelo tenga problemas para delimitar entidades correctamente y para distinguir entre `PER` y `LOC`.

Otra limitación es que el tokenizador BPE se entrena sobre un corpus pequeño. Por eso algunas palabras se segmentan de manera poco natural, especialmente nombres propios o palabras con mayúsculas.

Estas limitaciones son importantes, pero también forman parte del aprendizaje de la práctica: muestran qué ocurre cuando se intenta construir un sistema completo con pocos datos y recursos limitados.

---

## 12. Conclusiones

La práctica ha permitido construir un pipeline completo de NLP desde cero, pasando por todas las fases importantes: tokenización, modelo causal, generación, adaptación a NER, evaluación y empaquetado como herramienta ejecutable.

La parte más valiosa del trabajo ha sido comprobar cómo se conectan las piezas. El tokenizador BPE condiciona la entrada del Transformer; el Transformer aprende representaciones durante el entrenamiento causal; esas representaciones se reutilizan después para NER; y las métricas finales muestran hasta qué punto esa transferencia funciona en un escenario pequeño.

Los resultados no son perfectos, pero sí son coherentes con el planteamiento. El modelo causal genera texto con rasgos reconocibles del corpus, aunque con errores. El modelo NER detecta bastantes entidades, pero con muchos falsos positivos. En ambos casos, los fallos se explican por el tamaño del dataset, el desbalance y la capacidad limitada del modelo.

Desde el punto de vista de ingeniería, el proyecto final queda empaquetado en un CLI reproducible y en un wheel instalable. Esto era importante para que la entrega no fuese solo código suelto, sino una herramienta completa que se puede ejecutar, probar y evaluar.

En conclusión, el sistema cumple el objetivo principal de la práctica: implementar y entender un mini Transformer funcional, aplicarlo a generación de texto y reutilizarlo para una tarea NER real construida a partir del corpus anotado en la preentrega. Aunque el modelo es pequeño, el recorrido completo refleja bien las decisiones y dificultades que aparecen en un proyecto real de procesamiento del lenguaje natural.

---

## 13. Integrantes

- Marina Triviño de las Heras
- Carlota Salazar Martín
